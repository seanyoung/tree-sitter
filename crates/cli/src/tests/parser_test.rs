use std::{
    ops::ControlFlow,
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
    time::{self, Duration},
};

use tree_sitter::{
    Decode, IncludedRangesError, InputEdit, Language, LogType, ParseOptions, ParseState, Parser,
    Point, Range, Tree,
};
use tree_sitter_generate::load_grammar_file;
use tree_sitter_proc_macro::retry;

use super::helpers::{
    allocations,
    edits::ReadRecorder,
    fixtures::{get_language, get_test_language},
};
use crate::{
    fuzz::edits::Edit,
    parse::perform_edit,
    tests::{
        generate_parser,
        helpers::fixtures::{fixtures_dir, get_test_fixture_language},
        invert_edit,
    },
};

#[test]
fn test_parsing_simple_string() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let tree = parser
        .parse(
            "
        struct Stuff {}
        fn main() {}
    ",
            None,
        )
        .unwrap();

    let root_node = tree.root_node();
    assert_eq!(root_node.kind(), "source_file");

    assert_eq!(
        root_node.to_sexp(),
        concat!(
            "(source_file ",
            "(struct_item name: (type_identifier) body: (field_declaration_list)) ",
            "(function_item name: (identifier) parameters: (parameters) body: (block)))"
        )
    );

    let struct_node = root_node.child(0).unwrap();
    assert_eq!(struct_node.kind(), "struct_item");
}

/// A minimal, fully-controlled grammar used for snapshotting the parser/lexer
/// log stream. Keeping it tiny and self-contained makes the snapshot stable and
/// independent of changes to the bundled fixture grammars.
fn log_snapshot_language() -> Language {
    let (name, code) = generate_parser(
        r#"{
            "name": "log_snapshot",
            "extras": [{"type": "PATTERN", "value": "\\s"}],
            "rules": {
                "program": {"type": "SEQ", "members": [
                    {"type": "STRING", "value": "a"},
                    {"type": "STRING", "value": "b"}
                ]}
            }
        }"#,
    )
    .unwrap();
    get_test_language(&name, &code, None)
}

/// Parse `src` with a logger attached and return every emitted log line as
/// `"<type> <message>"`, one per line. This captures `parser.c`'s `LOG` /
/// `LOG_LOOKAHEAD` (`TSLogTypeParse`) and `lexer.c`'s `LOG` (`TSLogTypeLex`).
fn collect_parse_log(language: &Language, src: &str) -> String {
    let mut parser = Parser::new();
    parser.set_language(language).unwrap();
    let mut lines: Vec<String> = Vec::new();
    // SAFETY: the logger borrows `lines` and is only invoked during the `parse`
    // call below; the logger is removed before `lines` is read.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|log_type, message| {
            let kind = match log_type {
                LogType::Parse => "parse",
                LogType::Lex => "lex",
            };
            lines.push(format!("{kind} {message}"));
        })));
    }
    parser.parse(src, None).unwrap();
    parser.set_logger(None);
    lines.join("\n")
}

// Snapshots of the full parser/lexer log stream for fixed inputs. These pin the
// exact message text, ordering, and presence of the `LOG` / `LOG_LOOKAHEAD`
// calls in parser.c (TSLogTypeParse) and the `LOG` calls in lexer.c
// (TSLogTypeLex). Removing, reordering, or changing the format of any logged
// call that fires for these inputs will fail the corresponding assertion.
//
// To update after an intentional change: run the test, copy the `left` value
// from the assertion failure into the expected snapshot below.
#[test]
fn test_parser_log_snapshot_clean_parse() {
    let expected = "\
parse new_parse
parse process version:0, version_count:1, state:1, row:0, col:0
parse lex_internal state:0, row:0, column:0
lex consume character:'a'
parse lexed_lookahead sym:a, size:1
parse shift state:2
parse process version:0, version_count:1, state:2, row:0, col:1
parse lex_internal state:0, row:0, column:1
lex skip character:' '
lex consume character:'b'
parse lexed_lookahead sym:b, size:2
parse shift state:4
parse process version:0, version_count:1, state:4, row:0, col:3
parse lex_internal state:0, row:0, column:3
parse lexed_lookahead sym:end, size:0
parse reduce sym:program, child_count:2
parse accept
parse done";
    assert_eq!(collect_parse_log(&log_snapshot_language(), "a b"), expected);
}

#[test]
fn test_parser_log_snapshot_error_recovery() {
    let expected = "\
parse new_parse
parse process version:0, version_count:1, state:1, row:0, col:0
parse lex_internal state:0, row:0, column:0
lex consume character:'a'
parse lexed_lookahead sym:a, size:1
parse shift state:2
parse process version:0, version_count:1, state:2, row:0, col:1
parse lex_internal state:0, row:0, column:1
lex skip character:' '
parse lex_internal state:0, row:0, column:1
lex skip character:' '
parse skip_unrecognized_character
lex consume character:'c'
parse lex_internal state:0, row:0, column:3
parse lexed_lookahead sym:ERROR, size:2
parse detect_error lookahead:ERROR
parse resume version:0
parse skip_token symbol:ERROR
parse process version:0, version_count:1, state:0, row:0, col:3
parse lex_internal state:0, row:0, column:3
parse lexed_lookahead sym:end, size:0
parse recover_eof
parse done";
    assert_eq!(collect_parse_log(&log_snapshot_language(), "a c"), expected);
}

#[test]
fn test_parsing_with_logging() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let mut messages = Vec::new();
    // SAFETY: the logger borrows `messages` and is only invoked during the
    // `parse` call below while `messages` is in scope.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|log_type, message| {
            messages.push((log_type, message.to_string()));
        })));
    }

    parser
        .parse(
            "
        struct Stuff {}
        fn main() {}
    ",
            None,
        )
        .unwrap();

    assert!(messages.contains(&(
        LogType::Parse,
        "reduce sym:struct_item, child_count:3".to_string()
    )));
    assert!(messages.contains(&(LogType::Lex, "skip character:' '".to_string())));

    let mut row_starts_from_0 = false;
    for (_, m) in &messages {
        if m.contains("row:0") {
            row_starts_from_0 = true;
            break;
        }
    }
    assert!(row_starts_from_0);
}

#[test]
fn test_parsing_with_debug_graph_enabled() {
    use std::io::{BufRead, BufReader, Seek};

    let has_zero_indexed_row = |s: &str| s.contains("position: 0,");

    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();

    let mut debug_graph_file = tempfile::tempfile().unwrap();
    parser.print_dot_graphs(&debug_graph_file);
    parser.parse("const zero = 0", None).unwrap();

    debug_graph_file.rewind().unwrap();
    let log_reader = BufReader::new(debug_graph_file)
        .lines()
        .map(|l| l.expect("Failed to read line from graph log"));
    for line in log_reader {
        assert!(
            !has_zero_indexed_row(&line),
            "Graph log output includes zero-indexed row: {line}",
        );
    }
}

#[test]
fn test_parsing_with_custom_utf8_input() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let lines = &["pub fn foo() {", "  1", "}"];

    let tree = parser
        .parse_with_options(
            &mut |_, position| {
                let row = position.row;
                let column = position.column;
                if row < lines.len() {
                    if column < lines[row].len() {
                        &lines[row].as_bytes()[column..]
                    } else {
                        b"\n"
                    }
                } else {
                    &[]
                }
            },
            None,
            None,
        )
        .unwrap();

    let root = tree.root_node();
    assert_eq!(
        root.to_sexp(),
        concat!(
            "(source_file ",
            "(function_item ",
            "(visibility_modifier) ",
            "name: (identifier) ",
            "parameters: (parameters) ",
            "body: (block (integer_literal))))"
        )
    );
    assert_eq!(root.kind(), "source_file");
    assert!(!root.has_error());
    assert_eq!(root.child(0).unwrap().kind(), "function_item");
}

#[test]
fn test_parsing_with_custom_utf16le_input() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let lines = ["pub fn foo() {", "  1", "}"]
        .iter()
        .map(|s| s.encode_utf16().map(u16::to_le).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    let newline = [('\n' as u16).to_le()];

    let tree = parser
        .parse_utf16_le_with_options(
            &mut |_, position| {
                let row = position.row;
                let column = position.column;
                if row < lines.len() {
                    if column < lines[row].len() {
                        &lines[row][column..]
                    } else {
                        &newline
                    }
                } else {
                    &[]
                }
            },
            None,
            None,
        )
        .unwrap();

    let root = tree.root_node();
    assert_eq!(
        root.to_sexp(),
        "(source_file (function_item (visibility_modifier) name: (identifier) parameters: (parameters) body: (block (integer_literal))))"
    );
    assert_eq!(root.kind(), "source_file");
    assert!(!root.has_error());
    assert_eq!(root.child(0).unwrap().kind(), "function_item");
}

#[test]
fn test_parsing_with_custom_utf16_be_input() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let lines: Vec<Vec<u16>> = ["pub fn foo() {", "  1", "}"]
        .iter()
        .map(|s| s.encode_utf16().collect::<Vec<_>>())
        .map(|v| v.iter().map(|u| u.to_be()).collect())
        .collect();

    let newline = [('\n' as u16).to_be()];

    let tree = parser
        .parse_utf16_be_with_options(
            &mut |_, position| {
                let row = position.row;
                let column = position.column;
                if row < lines.len() {
                    if column < lines[row].len() {
                        &lines[row][column..]
                    } else {
                        &newline
                    }
                } else {
                    &[]
                }
            },
            None,
            None,
        )
        .unwrap();
    let root = tree.root_node();
    assert_eq!(
        root.to_sexp(),
        "(source_file (function_item (visibility_modifier) name: (identifier) parameters: (parameters) body: (block (integer_literal))))"
    );
    assert_eq!(root.kind(), "source_file");
    assert!(!root.has_error());
    assert_eq!(root.child(0).unwrap().kind(), "function_item");
}

#[test]
fn test_utf16_decodes_surrogate_pairs() {
    let mut parser = Parser::new();
    let language = get_test_fixture_language("utf16_surrogate_oob");
    parser.set_language(&language).unwrap();

    let le = [0xD83D_u16.to_le(), 0xDE00_u16.to_le()];
    let tree = parser.parse_utf16_le(le, None).unwrap();
    assert_eq!(tree.root_node().to_sexp(), "(program (supplementary))");

    let be = [0xD83D_u16.to_be(), 0xDE00_u16.to_be()];
    let tree = parser.parse_utf16_be(be, None).unwrap();
    assert_eq!(tree.root_node().to_sexp(), "(program (supplementary))");
}

#[test]
fn test_utf16_decode_does_not_read_oob() {
    // Test for a buffer over-read in ts_decode_utf16_le/be when a lead surrogate
    // is the last code unit in a chunk. The test grammar's external scanner
    // distinguishes surrogate code points from supplementary-plane characters,
    // making the over-read directly observable in the parse tree.
    //
    // Buffer layout:
    //   buf[0] = 0xD83E  (lead surrogate)
    //   buf[1] = 0xDD8B  (POISON: fake trail surrogate, adjacent in memory)
    //
    // The callback returns only buf[0..1] (one code unit = 2 bytes).
    //
    // When functioning correctly, this test passes a length of 2 bytes, which is
    // interpreted as 2/2 = 1 code unit, and thus doesn't over-read into the "poison"
    // fake trail surrogate. If an over-read does occur, the scanner sees a
    // supplementary token.
    let mut parser = Parser::new();
    let language = get_test_fixture_language("utf16_surrogate_oob");
    parser.set_language(&language).unwrap();

    let buf = vec![
        0xD83E, // lead surrogate (the only "visible" code unit)
        0xDD8B, // POISON: adjacent in Vec memory, past the chunk
    ];
    assert_eq!("🦋", String::from_utf16(&buf).unwrap());

    let mut callback = |offset: usize, _position: Point| -> &[u16] {
        // only expose buf[0], never buf[1]
        if offset >= 1 {
            return [].as_slice();
        }
        &buf[0..1]
    };

    // Use the parse function matching the host endianness, since the
    // buffer contains native u16 values.
    #[cfg(target_endian = "little")]
    let tree = parser
        .parse_utf16_le_with_options(&mut callback, None, None)
        .unwrap();
    #[cfg(target_endian = "big")]
    let tree = parser
        .parse_utf16_be_with_options(&mut callback, None, None)
        .unwrap();

    let root = tree.root_node();

    // Correct: scanner sees raw surrogate (0xD83E) -> `surrogate` node
    // Incorrect: scanner sees supplementary (U+1F98B, aka 🦋) -> `supplementary` node
    assert_eq!(
        root.to_sexp(),
        "(program (surrogate))",
        "buffer over-read: decoder read past chunk boundary and formed a \
         supplementary character from OOB adjacent memory"
    );
}

#[test]
fn test_parsing_with_callback_returning_owned_strings() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let text = b"pub fn foo() { 1 }";

    let tree = parser
        .parse_with_options(
            &mut |i, _| String::from_utf8(text[i..].to_vec()).unwrap(),
            None,
            None,
        )
        .unwrap();

    let root = tree.root_node();
    assert_eq!(
        root.to_sexp(),
        "(source_file (function_item (visibility_modifier) name: (identifier) parameters: (parameters) body: (block (integer_literal))))"
    );
}

#[test]
fn test_parsing_text_with_byte_order_mark() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    // Parse UTF16 text with a BOM
    let tree = parser
        .parse_utf16_le(
            "\u{FEFF}fn a() {}"
                .encode_utf16()
                .map(u16::to_le)
                .collect::<Vec<_>>(),
            None,
        )
        .unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (function_item name: (identifier) parameters: (parameters) body: (block)))"
    );
    assert_eq!(tree.root_node().start_byte(), 2);

    // Parse UTF8 text with a BOM
    let mut tree = parser.parse("\u{FEFF}fn a() {}", None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (function_item name: (identifier) parameters: (parameters) body: (block)))"
    );
    assert_eq!(tree.root_node().start_byte(), 3);

    // Edit the text, inserting a character before the BOM. The BOM is now an error.
    tree.edit(&InputEdit {
        start_byte: 0,
        old_end_byte: 0,
        new_end_byte: 1,
        start_position: Point::new(0, 0),
        old_end_position: Point::new(0, 0),
        new_end_position: Point::new(0, 1),
    });
    let mut tree = parser.parse(" \u{FEFF}fn a() {}", Some(&tree)).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (ERROR (UNEXPECTED 65279)) (function_item name: (identifier) parameters: (parameters) body: (block)))"
    );
    assert_eq!(tree.root_node().start_byte(), 1);

    // Edit the text again, putting the BOM back at the beginning.
    tree.edit(&InputEdit {
        start_byte: 0,
        old_end_byte: 1,
        new_end_byte: 0,
        start_position: Point::new(0, 0),
        old_end_position: Point::new(0, 1),
        new_end_position: Point::new(0, 0),
    });
    let tree = parser.parse("\u{FEFF}fn a() {}", Some(&tree)).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (function_item name: (identifier) parameters: (parameters) body: (block)))"
    );
    assert_eq!(tree.root_node().start_byte(), 3);
}

#[test]
fn test_parsing_invalid_chars_at_eof() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("json")).unwrap();
    let tree = parser.parse(b"\xdf", None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(document (ERROR (UNEXPECTED INVALID)))"
    );
}

#[test]
fn test_parsing_unexpected_null_characters_within_source() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();
    let tree = parser.parse(b"var \0 something;", None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(program (variable_declaration (ERROR (UNEXPECTED '\\0')) (variable_declarator name: (identifier))))"
    );
}

#[test]
fn test_parsing_ends_when_input_callback_returns_empty() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();
    let mut i = 0;
    let source = b"abcdefghijklmnoqrs";
    let tree = parser
        .parse_with_options(
            &mut |offset, _| {
                i += 1;
                if offset >= 6 {
                    b""
                } else {
                    &source[offset..usize::min(source.len(), offset + 3)]
                }
            },
            None,
            None,
        )
        .unwrap();
    assert_eq!(tree.root_node().end_byte(), 6);
}

// Incremental parsing

#[test]
fn test_parsing_after_editing_beginning_of_code() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();

    let mut code = b"123 + 456 * (10 + x);".to_vec();
    let mut tree = parser.parse(&code, None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(program (expression_statement (binary_expression ",
            "left: (number) ",
            "right: (binary_expression left: (number) right: (parenthesized_expression ",
            "(binary_expression left: (number) right: (identifier)))))))",
        )
    );

    perform_edit(
        &mut tree,
        &mut code,
        &Edit {
            position: 3,
            deleted_length: 0,
            inserted_text: b" || 5".to_vec(),
        },
    )
    .unwrap();

    let mut recorder = ReadRecorder::new(&code);
    let tree = parser
        .parse_with_options(&mut |i, _| recorder.read(i), Some(&tree), None)
        .unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(program (expression_statement (binary_expression ",
            "left: (number) ",
            "right: (binary_expression ",
            "left: (number) ",
            "right: (binary_expression ",
            "left: (number) ",
            "right: (parenthesized_expression (binary_expression left: (number) right: (identifier))))))))",
        )
    );

    assert_eq!(recorder.strings_read(), vec!["123 || 5 "]);
}

#[test]
fn test_parsing_after_editing_end_of_code() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();

    let mut code = b"x * (100 + abc);".to_vec();
    let mut tree = parser.parse(&code, None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(program (expression_statement (binary_expression ",
            "left: (identifier) ",
            "right: (parenthesized_expression (binary_expression left: (number) right: (identifier))))))",
        )
    );

    let position = code.len() - 2;
    perform_edit(
        &mut tree,
        &mut code,
        &Edit {
            position,
            deleted_length: 0,
            inserted_text: b".d".to_vec(),
        },
    )
    .unwrap();

    let mut recorder = ReadRecorder::new(&code);
    let tree = parser
        .parse_with_options(&mut |i, _| recorder.read(i), Some(&tree), None)
        .unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(program (expression_statement (binary_expression ",
            "left: (identifier) ",
            "right: (parenthesized_expression (binary_expression ",
            "left: (number) ",
            "right: (member_expression ",
            "object: (identifier) ",
            "property: (property_identifier)))))))"
        )
    );

    assert_eq!(recorder.strings_read(), vec![" * ", "abc.d)",]);
}

#[test]
fn test_parsing_empty_file_with_reused_tree() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let tree = parser.parse("", None);
    parser.parse("", tree.as_ref());

    let tree = parser.parse("\n  ", None);
    parser.parse("\n  ", tree.as_ref());
}

#[test]
fn test_parsing_after_editing_tree_that_depends_on_column_values() {
    let mut parser = Parser::new();
    parser
        .set_language(&get_test_fixture_language("uses_current_column"))
        .unwrap();

    let mut code = b"
a = b
c = do d
       e + f
       g
h + i
    "
    .to_vec();
    let mut tree = parser.parse(&code, None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(block ",
            "(binary_expression (identifier) (identifier)) ",
            "(binary_expression (identifier) (do_expression (block (identifier) (binary_expression (identifier) (identifier)) (identifier)))) ",
            "(binary_expression (identifier) (identifier)))",
        )
    );

    perform_edit(
        &mut tree,
        &mut code,
        &Edit {
            position: 8,
            deleted_length: 0,
            inserted_text: b"1234".to_vec(),
        },
    )
    .unwrap();

    assert_eq!(
        code,
        b"
a = b
c1234 = do d
       e + f
       g
h + i
    "
    );

    let mut recorder = ReadRecorder::new(&code);
    let tree = parser
        .parse_with_options(&mut |i, _| recorder.read(i), Some(&tree), None)
        .unwrap();

    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(block ",
            "(binary_expression (identifier) (identifier)) ",
            "(binary_expression (identifier) (do_expression (block (identifier)))) ",
            "(binary_expression (identifier) (identifier)) ",
            "(identifier) ",
            "(binary_expression (identifier) (identifier)))",
        )
    );

    assert_eq!(
        recorder.strings_read(),
        vec!["\nc1234 = do d\n       e + f\n       g\n"]
    );
}

#[test]
fn test_parsing_after_editing_tree_that_depends_on_column_position() {
    let mut parser = Parser::new();
    parser
        .set_language(&get_test_fixture_language("depends_on_column"))
        .unwrap();

    let mut code = b"\n x".to_vec();
    let mut tree = parser.parse(&code, None).unwrap();
    assert_eq!(tree.root_node().to_sexp(), "(x_is_at (odd_column))");

    perform_edit(
        &mut tree,
        &mut code,
        &Edit {
            position: 1,
            deleted_length: 0,
            inserted_text: b" ".to_vec(),
        },
    )
    .unwrap();

    assert_eq!(code, b"\n  x");

    let mut recorder = ReadRecorder::new(&code);
    let mut tree = parser
        .parse_with_options(&mut |i, _| recorder.read(i), Some(&tree), None)
        .unwrap();

    assert_eq!(tree.root_node().to_sexp(), "(x_is_at (even_column))",);
    assert_eq!(recorder.strings_read(), vec!["\n  x"]);

    perform_edit(
        &mut tree,
        &mut code,
        &Edit {
            position: 1,
            deleted_length: 0,
            inserted_text: b"\n".to_vec(),
        },
    )
    .unwrap();

    assert_eq!(code, b"\n\n  x");

    let mut recorder = ReadRecorder::new(&code);
    let tree = parser
        .parse_with_options(&mut |i, _| recorder.read(i), Some(&tree), None)
        .unwrap();

    assert_eq!(tree.root_node().to_sexp(), "(x_is_at (even_column))",);
    assert_eq!(recorder.strings_read(), vec!["\n\n  x"]);
}

#[test]
fn test_parsing_after_detecting_error_in_the_middle_of_a_string_token() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("python")).unwrap();

    let mut source = b"a = b, 'c, d'".to_vec();
    let tree = parser.parse(&source, None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(module (expression_statement (assignment left: (identifier) right: (expression_list (identifier) (string (string_start) (string_content) (string_end))))))"
    );

    // Delete a suffix of the source code, starting in the middle of the string
    // literal, after some whitespace. With this deletion, the remaining string
    // content: "c, " looks like two valid python tokens: an identifier and a comma.
    // When this edit is undone, in order correctly recover the original tree, the
    // parser needs to remember that before matching the `c` as an identifier, it
    // lookahead ahead several bytes, trying to find the closing quotation mark in
    // order to match the "string content" node.
    let edit_ix = std::str::from_utf8(&source).unwrap().find("d'").unwrap();
    let edit = Edit {
        position: edit_ix,
        deleted_length: source.len() - edit_ix,
        inserted_text: Vec::new(),
    };
    let undo = invert_edit(&source, &edit);

    let mut tree2 = tree.clone();
    perform_edit(&mut tree2, &mut source, &edit).unwrap();
    tree2 = parser.parse(&source, Some(&tree2)).unwrap();
    assert!(tree2.root_node().has_error());

    let mut tree3 = tree2.clone();
    perform_edit(&mut tree3, &mut source, &undo).unwrap();
    tree3 = parser.parse(&source, Some(&tree3)).unwrap();
    assert_eq!(tree3.root_node().to_sexp(), tree.root_node().to_sexp(),);
}

// Thread safety

#[test]
fn test_parsing_on_multiple_threads() {
    // Parse this source file so that each thread has a non-trivial amount of
    // work to do.
    let this_file_source = include_str!("parser_test.rs");

    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();
    let tree = parser.parse(this_file_source, None).unwrap();

    let mut parse_threads = Vec::new();
    for thread_id in 1..5 {
        let mut tree_clone = tree.clone();
        parse_threads.push(thread::spawn(move || {
            // For each thread, prepend a different number of declarations to the
            // source code.
            let mut prepend_line_count = 0;
            let mut prepended_source = String::new();
            for _ in 0..thread_id {
                prepend_line_count += 2;
                prepended_source += "struct X {}\n\n";
            }

            tree_clone.edit(&InputEdit {
                start_byte: 0,
                old_end_byte: 0,
                new_end_byte: prepended_source.len(),
                start_position: Point::new(0, 0),
                old_end_position: Point::new(0, 0),
                new_end_position: Point::new(prepend_line_count, 0),
            });
            prepended_source += this_file_source;

            // Reparse using the old tree as a starting point.
            let mut parser = Parser::new();
            parser.set_language(&get_language("rust")).unwrap();
            parser.parse(&prepended_source, Some(&tree_clone)).unwrap()
        }));
    }

    // Check that the trees have the expected relationship to one another.
    let trees = parse_threads
        .into_iter()
        .map(|thread| thread.join().unwrap());
    let child_count_differences = trees
        .map(|t| t.root_node().child_count() - tree.root_node().child_count())
        .collect::<Vec<_>>();

    assert_eq!(child_count_differences, &[1, 2, 3, 4]);
}

#[test]
fn test_parsing_cancelled_by_another_thread() {
    let cancellation_flag = std::sync::Arc::new(AtomicUsize::new(0));
    let flag = cancellation_flag.clone();
    let callback = &mut |_: &ParseState| {
        if cancellation_flag.load(Ordering::SeqCst) != 0 {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    };

    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();

    // Long input - parsing succeeds
    let tree = parser.parse_with_options(
        &mut |offset, _| {
            if offset == 0 {
                " [".as_bytes()
            } else if offset >= 20000 {
                "".as_bytes()
            } else {
                "0,".as_bytes()
            }
        },
        None,
        Some(ParseOptions::new().progress_callback(callback)),
    );
    assert!(tree.is_some());

    let cancel_thread = thread::spawn(move || {
        thread::sleep(time::Duration::from_millis(100));
        flag.store(1, Ordering::SeqCst);
    });

    // Infinite input
    let tree = parser.parse_with_options(
        &mut |offset, _| {
            thread::yield_now();
            thread::sleep(time::Duration::from_millis(10));
            if offset == 0 { b" [" } else { b"0," }
        },
        None,
        Some(ParseOptions::new().progress_callback(callback)),
    );

    // Parsing returns None because it was cancelled.
    cancel_thread.join().unwrap();
    assert!(tree.is_none());
}

// Timeouts

#[test]
#[retry(10)]
fn test_parsing_with_a_timeout() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("json")).unwrap();

    // Parse an infinitely-long array, but pause after 1ms of processing.
    let start_time = time::Instant::now();
    let tree = parser.parse_with_options(
        &mut |offset, _| {
            if offset == 0 { b" [" } else { b",0" }
        },
        None,
        Some(ParseOptions::new().progress_callback(&mut |_| {
            if start_time.elapsed().as_micros() > 1000 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })),
    );
    assert!(tree.is_none());
    assert!(start_time.elapsed().as_micros() < 2000);

    // Continue parsing, but pause after 1 ms of processing.
    let start_time = time::Instant::now();
    let tree = parser.parse_with_options(
        &mut |offset, _| {
            if offset == 0 { b" [" } else { b",0" }
        },
        None,
        Some(ParseOptions::new().progress_callback(&mut |_| {
            if start_time.elapsed().as_micros() > 5000 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })),
    );
    assert!(tree.is_none());
    assert!(start_time.elapsed().as_micros() > 100);
    assert!(start_time.elapsed().as_micros() < 10000);

    // Finish parsing
    let tree = parser
        .parse_with_options(
            &mut |offset, _| match offset {
                5001.. => "".as_bytes(),
                5000 => "]".as_bytes(),
                _ => ",0".as_bytes(),
            },
            None,
            None,
        )
        .unwrap();
    assert_eq!(tree.root_node().child(0).unwrap().kind(), "array");
}

#[test]
#[retry(10)]
fn test_parsing_with_a_timeout_and_a_reset() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("json")).unwrap();

    let start_time = time::Instant::now();
    let code = "[\"ok\", 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]";
    let tree = parser.parse_with_options(
        &mut |offset, _| {
            if offset >= code.len() {
                &[]
            } else {
                &code.as_bytes()[offset..]
            }
        },
        None,
        Some(ParseOptions::new().progress_callback(&mut |_| {
            if start_time.elapsed().as_micros() > 5 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })),
    );
    assert!(tree.is_none());

    // Without calling reset, the parser continues from where it left off, so
    // it does not see the changes to the beginning of the source code.
    let tree = parser.parse(
        "[null, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]",
        None,
    ).unwrap();
    assert_eq!(
        tree.root_node()
            .named_child(0)
            .unwrap()
            .named_child(0)
            .unwrap()
            .kind(),
        "string"
    );

    let start_time = time::Instant::now();
    let code = "[\"ok\", 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]";
    let tree = parser.parse_with_options(
        &mut |offset, _| {
            if offset >= code.len() {
                &[]
            } else {
                &code.as_bytes()[offset..]
            }
        },
        None,
        Some(ParseOptions::new().progress_callback(&mut |_| {
            if start_time.elapsed().as_micros() > 5 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })),
    );
    assert!(tree.is_none());

    // By calling reset, we force the parser to start over from scratch so
    // that it sees the changes to the beginning of the source code.
    parser.reset();
    let tree = parser.parse(
        "[null, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]",
        None,
    ).unwrap();
    assert_eq!(
        tree.root_node()
            .named_child(0)
            .unwrap()
            .named_child(0)
            .unwrap()
            .kind(),
        "null"
    );
}

#[test]
#[retry(10)]
fn test_parsing_with_a_timeout_and_implicit_reset() {
    allocations::record(|| {
        let mut parser = Parser::new();
        parser.set_language(&get_language("javascript")).unwrap();

        let code = "[\"ok\", 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]";
        let start_time = time::Instant::now();
        let tree = parser.parse_with_options(
            &mut |offset, _| {
                if offset >= code.len() {
                    &[]
                } else {
                    &code.as_bytes()[offset..]
                }
            },
            None,
            Some(ParseOptions::new().progress_callback(&mut |_| {
                if start_time.elapsed().as_micros() > 5 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })),
        );
        assert!(tree.is_none());

        // Changing the parser's language implicitly resets, discarding
        // the previous partial parse.
        parser.set_language(&get_language("json")).unwrap();
        let tree = parser.parse(
            "[null, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]",
            None,
        ).unwrap();
        assert_eq!(
            tree.root_node()
                .named_child(0)
                .unwrap()
                .named_child(0)
                .unwrap()
                .kind(),
            "null"
        );
    });
}

#[test]
#[retry(10)]
fn test_parsing_with_timeout_and_no_completion() {
    allocations::record(|| {
        let mut parser = Parser::new();
        parser.set_language(&get_language("javascript")).unwrap();

        let code = "[\"ok\", 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]";
        let start_time = time::Instant::now();
        let tree = parser.parse_with_options(
            &mut |offset, _| {
                if offset >= code.len() {
                    &[]
                } else {
                    &code.as_bytes()[offset..]
                }
            },
            None,
            Some(ParseOptions::new().progress_callback(&mut |_| {
                if start_time.elapsed().as_micros() > 5 {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })),
        );
        assert!(tree.is_none());

        // drop the parser when it has an unfinished parse
    });
}

#[test]
fn test_parsing_with_timeout_during_balancing() {
    allocations::record(|| {
        let mut parser = Parser::new();
        parser.set_language(&get_language("javascript")).unwrap();

        let function_count: u32 = 100;

        let code = "function() {}\n".repeat(function_count as usize);
        let mut current_byte_offset = 0;
        let mut in_balancing = false;
        let tree = parser.parse_with_options(
            &mut |offset, _| {
                if offset >= code.len() {
                    &[]
                } else {
                    &code.as_bytes()[offset..]
                }
            },
            None,
            Some(ParseOptions::new().progress_callback(&mut |state| {
                // The parser will call the progress_callback during parsing, and at the very end
                // during tree-balancing. For very large trees, this balancing act can take quite
                // some time, so we want to verify that timing out during this operation is
                // possible.
                //
                // We verify this by checking the current byte offset, as this number will *not* be
                // updated during tree balancing. If we see the same offset twice, we know that we
                // are in the balancing phase.
                if state.current_byte_offset() != current_byte_offset {
                    current_byte_offset = state.current_byte_offset();
                    ControlFlow::Continue(())
                } else {
                    in_balancing = true;
                    ControlFlow::Break(())
                }
            })),
        );

        assert!(tree.is_none());
        assert!(in_balancing);

        // This should not cause an assertion failure.
        parser.reset();
        let tree = parser.parse_with_options(
            &mut |offset, _| {
                if offset >= code.len() {
                    &[]
                } else {
                    &code.as_bytes()[offset..]
                }
            },
            None,
            Some(ParseOptions::new().progress_callback(&mut |state| {
                if state.current_byte_offset() != current_byte_offset {
                    current_byte_offset = state.current_byte_offset();
                    ControlFlow::Continue(())
                } else {
                    in_balancing = true;
                    ControlFlow::Break(())
                }
            })),
        );

        assert!(tree.is_none());
        assert!(in_balancing);

        // If we resume parsing (implying we didn't call `parser.reset()`), we should be able to
        // finish parsing the tree, continuing from where we left off.
        let tree = parser
            .parse_with_options(
                &mut |offset, _| {
                    if offset >= code.len() {
                        &[]
                    } else {
                        &code.as_bytes()[offset..]
                    }
                },
                None,
                Some(ParseOptions::new().progress_callback(&mut |state| {
                    // Because we've already finished parsing, we should only be resuming the
                    // balancing phase.
                    assert_eq!(state.current_byte_offset(), current_byte_offset);
                    ControlFlow::Continue(())
                })),
            )
            .unwrap();
        assert!(!tree.root_node().has_error());
        assert_eq!(tree.root_node().child_count(), function_count);
    });
}

#[test]
fn test_parsing_with_timeout_when_error_detected() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("json")).unwrap();

    // Parse an infinitely-long array, but insert an error after 1000 characters.
    let mut offset = 0;
    let erroneous_code = "!,";
    let tree = parser.parse_with_options(
        &mut |i, _| match i {
            0 => "[",
            1..=1000 => "0,",
            _ => erroneous_code,
        },
        None,
        Some(ParseOptions::new().progress_callback(&mut |state| {
            offset = state.current_byte_offset();
            if state.has_error() {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })),
    );

    // The callback is called at the end of parsing, however, what we're asserting here is that
    // parsing ends immediately as the error is detected. This is verified by checking the offset
    // of the last byte processed is the length of the erroneous code we inserted, aka, 1002, or
    // 1000 + the length of the erroneous code.
    assert_eq!(offset, 1000 + erroneous_code.len());
    assert!(tree.is_none());
}

// Included Ranges

#[test]
fn test_parsing_with_one_included_range() {
    let source_code = "<span>hi</span><script>console.log('sup');</script>";

    let mut parser = Parser::new();
    parser.set_language(&get_language("html")).unwrap();
    let html_tree = parser.parse(source_code, None).unwrap();
    let script_content_node = html_tree.root_node().child(1).unwrap().child(1).unwrap();
    assert_eq!(script_content_node.kind(), "raw_text");

    assert_eq!(
        parser.included_ranges(),
        &[Range {
            start_byte: 0,
            end_byte: u32::MAX as usize,
            start_point: Point::new(0, 0),
            end_point: Point::new(u32::MAX as usize, u32::MAX as usize),
        }]
    );
    parser
        .set_included_ranges(&[script_content_node.range()])
        .unwrap();
    assert_eq!(parser.included_ranges(), &[script_content_node.range()]);
    parser.set_language(&get_language("javascript")).unwrap();
    let js_tree = parser.parse(source_code, None).unwrap();

    assert_eq!(
        js_tree.root_node().to_sexp(),
        concat!(
            "(program (expression_statement (call_expression ",
            "function: (member_expression object: (identifier) property: (property_identifier)) ",
            "arguments: (arguments (string (string_fragment))))))",
        )
    );
    assert_eq!(
        js_tree.root_node().start_position(),
        Point::new(0, source_code.find("console").unwrap())
    );
    assert_eq!(js_tree.included_ranges(), &[script_content_node.range()]);
}

#[test]
fn test_parsing_with_multiple_included_ranges() {
    let source_code = "html `<div>Hello, ${name.toUpperCase()}, it's <b>${now()}</b>.</div>`";

    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();
    let js_tree = parser.parse(source_code, None).unwrap();
    let template_string_node = js_tree
        .root_node()
        .descendant_for_byte_range(
            source_code.find("`<").unwrap(),
            source_code.find(">`").unwrap(),
        )
        .unwrap();
    assert_eq!(template_string_node.kind(), "template_string");

    let open_quote_node = template_string_node.child(0).unwrap();
    let interpolation_node1 = template_string_node.child(2).unwrap();
    let interpolation_node2 = template_string_node.child(4).unwrap();
    let close_quote_node = template_string_node.child(6).unwrap();

    parser.set_language(&get_language("html")).unwrap();
    let html_ranges = &[
        Range {
            start_byte: open_quote_node.end_byte(),
            start_point: open_quote_node.end_position(),
            end_byte: interpolation_node1.start_byte(),
            end_point: interpolation_node1.start_position(),
        },
        Range {
            start_byte: interpolation_node1.end_byte(),
            start_point: interpolation_node1.end_position(),
            end_byte: interpolation_node2.start_byte(),
            end_point: interpolation_node2.start_position(),
        },
        Range {
            start_byte: interpolation_node2.end_byte(),
            start_point: interpolation_node2.end_position(),
            end_byte: close_quote_node.start_byte(),
            end_point: close_quote_node.start_position(),
        },
    ];
    parser.set_included_ranges(html_ranges).unwrap();
    let html_tree = parser.parse(source_code, None).unwrap();

    assert_eq!(
        html_tree.root_node().to_sexp(),
        concat!(
            "(document (element",
            " (start_tag (tag_name))",
            " (text)",
            " (element (start_tag (tag_name)) (end_tag (tag_name)))",
            " (text)",
            " (end_tag (tag_name))))",
        )
    );
    assert_eq!(html_tree.included_ranges(), html_ranges);

    let div_element_node = html_tree.root_node().child(0).unwrap();
    let hello_text_node = div_element_node.child(1).unwrap();
    let b_element_node = div_element_node.child(2).unwrap();
    let b_start_tag_node = b_element_node.child(0).unwrap();
    let b_end_tag_node = b_element_node.child(1).unwrap();

    assert_eq!(hello_text_node.kind(), "text");
    assert_eq!(
        hello_text_node.start_byte(),
        source_code.find("Hello").unwrap()
    );
    assert_eq!(
        hello_text_node.end_byte(),
        source_code.find(" <b>").unwrap()
    );

    assert_eq!(b_start_tag_node.kind(), "start_tag");
    assert_eq!(
        b_start_tag_node.start_byte(),
        source_code.find("<b>").unwrap()
    );
    assert_eq!(
        b_start_tag_node.end_byte(),
        source_code.find("${now()}").unwrap()
    );

    assert_eq!(b_end_tag_node.kind(), "end_tag");
    assert_eq!(
        b_end_tag_node.start_byte(),
        source_code.find("</b>").unwrap()
    );
    assert_eq!(
        b_end_tag_node.end_byte(),
        source_code.find(".</div>").unwrap()
    );
}

#[test]
fn test_parsing_with_included_range_containing_mismatched_positions() {
    let source_code = "<div>test</div>{_ignore_this_part_}";

    let mut parser = Parser::new();
    parser.set_language(&get_language("html")).unwrap();

    let end_byte = source_code.find("{_ignore_this_part_").unwrap();

    let range_to_parse = Range {
        start_byte: 0,
        start_point: Point {
            row: 10,
            column: 12,
        },
        end_byte,
        end_point: Point {
            row: 10,
            column: 12 + end_byte,
        },
    };

    parser.set_included_ranges(&[range_to_parse]).unwrap();

    let html_tree = parser
        .parse_with_options(&mut chunked_input(source_code, 3), None, None)
        .unwrap();

    assert_eq!(html_tree.root_node().range(), range_to_parse);

    assert_eq!(
        html_tree.root_node().to_sexp(),
        "(document (element (start_tag (tag_name)) (text) (end_tag (tag_name))))"
    );
}

#[test]
fn test_parsing_error_in_invalid_included_ranges() {
    let mut parser = Parser::new();

    // Ranges are not ordered
    let error = parser
        .set_included_ranges(&[
            Range {
                start_byte: 23,
                end_byte: 29,
                start_point: Point::new(0, 23),
                end_point: Point::new(0, 29),
            },
            Range {
                start_byte: 0,
                end_byte: 5,
                start_point: Point::new(0, 0),
                end_point: Point::new(0, 5),
            },
            Range {
                start_byte: 50,
                end_byte: 60,
                start_point: Point::new(0, 50),
                end_point: Point::new(0, 60),
            },
        ])
        .unwrap_err();
    assert_eq!(error, IncludedRangesError(1));

    // Range ends before it starts
    let error = parser
        .set_included_ranges(&[Range {
            start_byte: 10,
            end_byte: 5,
            start_point: Point::new(0, 10),
            end_point: Point::new(0, 5),
        }])
        .unwrap_err();
    assert_eq!(error, IncludedRangesError(0));
}

#[test]
fn test_parsing_utf16_code_with_errors_at_the_end_of_an_included_range() {
    let source_code = "<script>a.</script>";
    let utf16_source_code = source_code
        .encode_utf16()
        .map(u16::to_le)
        .collect::<Vec<_>>();

    let start_byte = 2 * source_code.find("a.").unwrap();
    let end_byte = 2 * source_code.find("</script>").unwrap();

    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();
    parser
        .set_included_ranges(&[Range {
            start_byte,
            end_byte,
            start_point: Point::new(0, start_byte),
            end_point: Point::new(0, end_byte),
        }])
        .unwrap();
    let tree = parser.parse_utf16_le(&utf16_source_code, None).unwrap();
    assert_eq!(tree.root_node().to_sexp(), "(program (ERROR (identifier)))");
}

#[test]
fn test_parsing_with_external_scanner_that_uses_included_range_boundaries() {
    let source_code = "a <%= b() %> c <% d() %>";
    let range1_start_byte = source_code.find(" b() ").unwrap();
    let range1_end_byte = range1_start_byte + " b() ".len();
    let range2_start_byte = source_code.find(" d() ").unwrap();
    let range2_end_byte = range2_start_byte + " d() ".len();

    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();
    parser
        .set_included_ranges(&[
            Range {
                start_byte: range1_start_byte,
                end_byte: range1_end_byte,
                start_point: Point::new(0, range1_start_byte),
                end_point: Point::new(0, range1_end_byte),
            },
            Range {
                start_byte: range2_start_byte,
                end_byte: range2_end_byte,
                start_point: Point::new(0, range2_start_byte),
                end_point: Point::new(0, range2_end_byte),
            },
        ])
        .unwrap();

    let tree = parser.parse(source_code, None).unwrap();
    let root = tree.root_node();
    let statement1 = root.child(0).unwrap();
    let statement2 = root.child(1).unwrap();

    assert_eq!(
        root.to_sexp(),
        concat!(
            "(program",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments)))",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments))))"
        )
    );

    assert_eq!(statement1.start_byte(), source_code.find("b()").unwrap());
    assert_eq!(statement1.end_byte(), source_code.find(" %> c").unwrap());
    assert_eq!(statement2.start_byte(), source_code.find("d()").unwrap());
    assert_eq!(statement2.end_byte(), source_code.len() - " %>".len());
}

#[test]
fn test_parsing_with_a_newly_excluded_range() {
    let mut source_code = String::from("<div><span><%= something %></span></div>");

    // Parse HTML including the template directive, which will cause an error
    let mut parser = Parser::new();
    parser.set_language(&get_language("html")).unwrap();
    let mut first_tree = parser
        .parse_with_options(&mut chunked_input(&source_code, 3), None, None)
        .unwrap();

    // Insert code at the beginning of the document.
    let prefix = "a very very long line of plain text. ";
    first_tree.edit(&InputEdit {
        start_byte: 0,
        old_end_byte: 0,
        new_end_byte: prefix.len(),
        start_position: Point::new(0, 0),
        old_end_position: Point::new(0, 0),
        new_end_position: Point::new(0, prefix.len()),
    });
    source_code.insert_str(0, prefix);

    // Parse the HTML again, this time *excluding* the template directive
    // (which has moved since the previous parse).
    let directive_start = source_code.find("<%=").unwrap();
    let directive_end = source_code.find("</span>").unwrap();
    let source_code_end = source_code.len();
    parser
        .set_included_ranges(&[
            Range {
                start_byte: 0,
                end_byte: directive_start,
                start_point: Point::new(0, 0),
                end_point: Point::new(0, directive_start),
            },
            Range {
                start_byte: directive_end,
                end_byte: source_code_end,
                start_point: Point::new(0, directive_end),
                end_point: Point::new(0, source_code_end),
            },
        ])
        .unwrap();
    let tree = parser
        .parse_with_options(&mut chunked_input(&source_code, 3), Some(&first_tree), None)
        .unwrap();

    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(document (text) (element",
            " (start_tag (tag_name))",
            " (element (start_tag (tag_name)) (end_tag (tag_name)))",
            " (end_tag (tag_name))))"
        )
    );

    assert_eq!(
        tree.changed_ranges(&first_tree).collect::<Vec<_>>(),
        vec![
            // The first range that has changed syntax is the range of the newly-inserted text.
            Range {
                start_byte: 0,
                end_byte: prefix.len(),
                start_point: Point::new(0, 0),
                end_point: Point::new(0, prefix.len()),
            },
            // Even though no edits were applied to the outer `div` element,
            // its contents have changed syntax because a range of text that
            // was previously included is now excluded.
            Range {
                start_byte: directive_start,
                end_byte: directive_end,
                start_point: Point::new(0, directive_start),
                end_point: Point::new(0, directive_end),
            },
        ]
    );
}

#[test]
fn test_parsing_with_a_newly_included_range() {
    let source_code = "<div><%= foo() %></div><span><%= bar() %></span><%= baz() %>";
    let range1_start = source_code.find(" foo").unwrap();
    let range2_start = source_code.find(" bar").unwrap();
    let range3_start = source_code.find(" baz").unwrap();
    let range1_end = range1_start + 7;
    let range2_end = range2_start + 7;
    let range3_end = range3_start + 7;

    // Parse only the first code directive as JavaScript
    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();
    parser
        .set_included_ranges(&[simple_range(range1_start, range1_end)])
        .unwrap();
    let tree = parser
        .parse_with_options(&mut chunked_input(source_code, 3), None, None)
        .unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        concat!(
            "(program",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments))))",
        )
    );

    // Parse both the first and third code directives as JavaScript, using the old tree as a
    // reference.
    parser
        .set_included_ranges(&[
            simple_range(range1_start, range1_end),
            simple_range(range3_start, range3_end),
        ])
        .unwrap();
    let tree2 = parser
        .parse_with_options(&mut chunked_input(source_code, 3), Some(&tree), None)
        .unwrap();
    assert_eq!(
        tree2.root_node().to_sexp(),
        concat!(
            "(program",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments)))",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments))))",
        )
    );
    assert_eq!(
        tree2.changed_ranges(&tree).collect::<Vec<_>>(),
        &[simple_range(range1_end, range3_end)]
    );

    // Parse all three code directives as JavaScript, using the old tree as a
    // reference.
    parser
        .set_included_ranges(&[
            simple_range(range1_start, range1_end),
            simple_range(range2_start, range2_end),
            simple_range(range3_start, range3_end),
        ])
        .unwrap();
    let tree3 = parser.parse(source_code, Some(&tree)).unwrap();
    assert_eq!(
        tree3.root_node().to_sexp(),
        concat!(
            "(program",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments)))",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments)))",
            " (expression_statement (call_expression function: (identifier) arguments: (arguments))))",
        )
    );
    assert_eq!(
        tree3.changed_ranges(&tree2).collect::<Vec<_>>(),
        &[simple_range(range2_start + 1, range2_end - 1)]
    );
}

#[test]
fn test_parsing_with_included_ranges_and_missing_tokens() {
    let (parser_name, parser_code) = generate_parser(
        r#"{
            "name": "test_leading_missing_token",
            "rules": {
                "program": {
                    "type": "SEQ",
                    "members": [
                        {"type": "SYMBOL", "name": "A"},
                        {"type": "SYMBOL", "name": "b"},
                        {"type": "SYMBOL", "name": "c"},
                        {"type": "SYMBOL", "name": "A"},
                        {"type": "SYMBOL", "name": "b"},
                        {"type": "SYMBOL", "name": "c"}
                    ]
                },
                "A": {"type": "SYMBOL", "name": "a"},
                "a": {"type": "STRING", "value": "a"},
                "b": {"type": "STRING", "value": "b"},
                "c": {"type": "STRING", "value": "c"}
            }
        }"#,
    )
    .unwrap();

    let mut parser = Parser::new();
    parser
        .set_language(&get_test_language(&parser_name, &parser_code, None))
        .unwrap();

    // There's a missing `a` token at the beginning of the code. It must be inserted
    // at the beginning of the first included range, not at {0, 0}.
    let source_code = "__bc__bc__";
    parser
        .set_included_ranges(&[
            Range {
                start_byte: 2,
                end_byte: 4,
                start_point: Point::new(0, 2),
                end_point: Point::new(0, 4),
            },
            Range {
                start_byte: 6,
                end_byte: 8,
                start_point: Point::new(0, 6),
                end_point: Point::new(0, 8),
            },
        ])
        .unwrap();

    let tree = parser.parse(source_code, None).unwrap();
    let root = tree.root_node();
    assert_eq!(
        root.to_sexp(),
        "(program (A (MISSING a)) (b) (c) (A (MISSING a)) (b) (c))"
    );
    assert_eq!(root.start_byte(), 2);
    assert_eq!(root.child(3).unwrap().start_byte(), 4);
}

#[test]
fn test_grammars_that_can_hang_on_eof() {
    let (parser_name, parser_code) = generate_parser(
        r#"
        {
            "name": "test_single_null_char_regex",
            "rules": {
                "source_file": {
                    "type": "SEQ",
                    "members": [
                        { "type": "STRING", "value": "\"" },
                        { "type": "PATTERN", "value": "[\\x00]*" },
                        { "type": "STRING", "value": "\"" }
                    ]
                }
            },
            "extras": [ { "type": "PATTERN", "value": "\\s" } ]
        }
        "#,
    )
    .unwrap();

    let mut parser = Parser::new();
    parser
        .set_language(&get_test_language(&parser_name, &parser_code, None))
        .unwrap();
    parser.parse("\"", None).unwrap();

    let (parser_name, parser_code) = generate_parser(
        r#"
        {
            "name": "test_null_char_with_next_char_regex",
            "rules": {
                "source_file": {
                    "type": "SEQ",
                    "members": [
                        { "type": "STRING", "value": "\"" },
                        { "type": "PATTERN", "value": "[\\x00-\\x01]*" },
                        { "type": "STRING", "value": "\"" }
                    ]
                }
            },
            "extras": [ { "type": "PATTERN", "value": "\\s" } ]
        }
        "#,
    )
    .unwrap();

    parser
        .set_language(&get_test_language(&parser_name, &parser_code, None))
        .unwrap();
    parser.parse("\"", None).unwrap();

    let (parser_name, parser_code) = generate_parser(
        r#"
        {
            "name": "test_null_char_with_range_regex",
            "rules": {
                "source_file": {
                    "type": "SEQ",
                    "members": [
                        { "type": "STRING", "value": "\"" },
                        { "type": "PATTERN", "value": "[\\x00-\\x7F]*" },
                        { "type": "STRING", "value": "\"" }
                    ]
                }
            },
            "extras": [ { "type": "PATTERN", "value": "\\s" } ]
        }
        "#,
    )
    .unwrap();

    parser
        .set_language(&get_test_language(&parser_name, &parser_code, None))
        .unwrap();
    parser.parse("\"", None).unwrap();
}

#[test]
fn test_parse_stack_recursive_merge_error_cost_calculation_bug() {
    let source_code = r"
fn main() {
  if n == 1 {
  } else if n == 2 {
  } else {
  }
}

let y = if x == 5 { 10 } else { 15 };

if foo && bar {}

if foo && bar || baz {}
";

    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let mut tree = parser.parse(source_code, None).unwrap();

    let edit = Edit {
        position: 60,
        deleted_length: 63,
        inserted_text: Vec::new(),
    };
    let mut input = source_code.as_bytes().to_vec();
    perform_edit(&mut tree, &mut input, &edit).unwrap();

    parser.parse(&input, Some(&tree)).unwrap();
}

#[test]
fn test_parsing_with_scanner_logging() {
    let mut parser = Parser::new();
    parser
        .set_language(&get_test_fixture_language("external_tokens"))
        .unwrap();

    let mut found = false;
    // SAFETY: the logger borrows `found` and is only invoked during the `parse`
    // call below, while `found` is in scope.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|log_type, message| {
            if log_type == LogType::Lex && message == "Found a percent string" {
                found = true;
            }
        })));
    }

    let source_code = "x + %(sup (external) scanner?)";

    parser.parse(source_code, None).unwrap();
    assert!(found);
}

#[test]
fn test_parsing_get_column_at_eof() {
    let mut parser = Parser::new();
    parser
        .set_language(&get_test_fixture_language("get_col_eof"))
        .unwrap();

    parser.parse("a", None).unwrap();
}

#[test]
fn test_parsing_by_halting_at_offset() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("javascript")).unwrap();

    let source_code = "function foo() { return 1; }".repeat(1000);

    let mut seen_byte_offsets = vec![];

    parser
        .parse_with_options(
            &mut |offset, _| {
                if offset < source_code.len() {
                    &source_code.as_bytes()[offset..]
                } else {
                    &[]
                }
            },
            None,
            Some(ParseOptions::new().progress_callback(&mut |p| {
                seen_byte_offsets.push(p.current_byte_offset());
                ControlFlow::Continue(())
            })),
        )
        .unwrap();

    assert!(seen_byte_offsets.len() > 100);
}

#[test]
fn test_decode_utf32() {
    use widestring::u32cstr;

    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let utf32_text = u32cstr!("pub fn foo() { println!(\"€50\"); }");
    let utf32_text = unsafe {
        std::slice::from_raw_parts(utf32_text.as_ptr().cast::<u8>(), utf32_text.len() * 4)
    };

    struct U32Decoder;

    impl Decode for U32Decoder {
        fn decode(bytes: &[u8]) -> (i32, u32) {
            if bytes.len() >= 4 {
                #[cfg(target_endian = "big")]
                {
                    (
                        i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                        4,
                    )
                }

                #[cfg(target_endian = "little")]
                {
                    (
                        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                        4,
                    )
                }
            } else {
                (0, 0)
            }
        }
    }

    let tree = parser
        .parse_custom_encoding::<U32Decoder, _, _>(
            &mut |offset, _| {
                if offset < utf32_text.len() {
                    &utf32_text[offset..]
                } else {
                    &[]
                }
            },
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (function_item (visibility_modifier) name: (identifier) parameters: (parameters) body: (block (expression_statement (macro_invocation macro: (identifier) (token_tree (string_literal (string_content))))))))"
    );
}

#[test]
fn test_decode_cp1252() {
    use encoding_rs::WINDOWS_1252;

    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let windows_1252_text = WINDOWS_1252.encode("pub fn foo() { println!(\"€50\"); }").0;

    struct Cp1252Decoder;

    impl Decode for Cp1252Decoder {
        fn decode(bytes: &[u8]) -> (i32, u32) {
            if !bytes.is_empty() {
                let byte = bytes[0];
                (i32::from(byte), 1)
            } else {
                (0, 0)
            }
        }
    }

    let tree = parser
        .parse_custom_encoding::<Cp1252Decoder, _, _>(
            &mut |offset, _| &windows_1252_text[offset..],
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (function_item (visibility_modifier) name: (identifier) parameters: (parameters) body: (block (expression_statement (macro_invocation macro: (identifier) (token_tree (string_literal (string_content))))))))"
    );
}

#[test]
fn test_decode_macintosh() {
    use encoding_rs::MACINTOSH;

    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let macintosh_text = MACINTOSH.encode("pub fn foo() { println!(\"€50\"); }").0;

    struct MacintoshDecoder;

    impl Decode for MacintoshDecoder {
        fn decode(bytes: &[u8]) -> (i32, u32) {
            if !bytes.is_empty() {
                let byte = bytes[0];
                (i32::from(byte), 1)
            } else {
                (0, 0)
            }
        }
    }

    let tree = parser
        .parse_custom_encoding::<MacintoshDecoder, _, _>(
            &mut |offset, _| &macintosh_text[offset..],
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (function_item (visibility_modifier) name: (identifier) parameters: (parameters) body: (block (expression_statement (macro_invocation macro: (identifier) (token_tree (string_literal (string_content))))))))"
    );
}

#[test]
fn test_decode_utf24le() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let mut utf24le_text = Vec::new();
    for c in "pub fn foo() { println!(\"€50\"); }".chars() {
        let code_point = c as u32;
        utf24le_text.push((code_point & 0xFF) as u8);
        utf24le_text.push(((code_point >> 8) & 0xFF) as u8);
        utf24le_text.push(((code_point >> 16) & 0xFF) as u8);
    }

    struct Utf24LeDecoder;

    impl Decode for Utf24LeDecoder {
        fn decode(bytes: &[u8]) -> (i32, u32) {
            if bytes.len() >= 3 {
                (i32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]), 3)
            } else {
                (0, 0)
            }
        }
    }

    let tree = parser
        .parse_custom_encoding::<Utf24LeDecoder, _, _>(
            &mut |offset, _| &utf24le_text[offset..],
            None,
            None,
        )
        .unwrap();

    assert_eq!(
        tree.root_node().to_sexp(),
        "(source_file (function_item (visibility_modifier) name: (identifier) parameters: (parameters) body: (block (expression_statement (macro_invocation macro: (identifier) (token_tree (string_literal (string_content))))))))"
    );
}

#[test]
fn test_grammars_that_should_not_compile() {
    assert!(
        generate_parser(
            r#"
        {
            "name": "issue_1111",
            "rules": {
                "source_file": { "type": "STRING", "value": "" }
            },
        }
        "#
        )
        .is_err()
    );

    assert!(
        generate_parser(
            r#"
        {
            "name": "issue_1271",
            "rules": {
                "source_file": { "type": "SYMBOL", "name": "identifier" },
                "identifier": {
                    "type": "TOKEN",
                    "content": {
                        "type": "REPEAT",
                        "content": { "type": "PATTERN", "value": "a" }
                    }
                }
            },
        }
        "#
        )
        .is_err()
    );

    assert!(
        generate_parser(
            r#"
        {
            "name": "issue_1156_expl_1",
            "rules": {
                "source_file": {
                    "type": "TOKEN",
                    "content": {
                        "type": "REPEAT",
                        "content": { "type": "STRING", "value": "c" }
                    }
                }
            },
        }
        "#
        )
        .is_err()
    );

    assert!(
        generate_parser(
            r#"
        {
            "name": "issue_1156_expl_2",
            "rules": {
                "source_file": {
                    "type": "TOKEN",
                    "content": {
                        "type": "CHOICE",
                        "members": [
                            { "type": "STRING", "value": "e" },
                            { "type": "BLANK" }
                        ]
                    }
                }
            },
        }
        "#
        )
        .is_err()
    );

    assert!(
        generate_parser(
            r#"
        {
            "name": "issue_1156_expl_3",
            "rules": {
                "source_file": {
                    "type": "IMMEDIATE_TOKEN",
                    "content": {
                        "type": "REPEAT",
                        "content": { "type": "STRING", "value": "p" }
                    }
                }
            },
        }
        "#
        )
        .is_err()
    );

    assert!(
        generate_parser(
            r#"
        {
            "name": "issue_1156_expl_4",
            "rules": {
                "source_file": {
                    "type": "IMMEDIATE_TOKEN",
                    "content": {
                        "type": "CHOICE",
                        "members": [
                            { "type": "STRING", "value": "r" },
                            { "type": "BLANK" }
                        ]
                    }
                }
            },
        }
        "#
        )
        .is_err()
    );
}

const fn simple_range(start: usize, end: usize) -> Range {
    Range {
        start_byte: start,
        end_byte: end,
        start_point: Point::new(0, start),
        end_point: Point::new(0, end),
    }
}

fn chunked_input<'a>(text: &'a str, size: usize) -> impl FnMut(usize, Point) -> &'a [u8] {
    move |offset, _| &text.as_bytes()[offset..text.len().min(offset + size)]
}

#[test]
fn test_parse_options_reborrow() {
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let parse_count = AtomicUsize::new(0);

    let mut callback = |_: &ParseState| {
        parse_count.fetch_add(1, Ordering::SeqCst);
        ControlFlow::Continue(())
    };
    let mut options = ParseOptions::new().progress_callback(&mut callback);

    let text1 = "fn first() {}".repeat(20);
    let text2 = "fn second() {}".repeat(20);

    let tree1 = parser
        .parse_with_options(
            &mut |offset, _| {
                if offset >= text1.len() {
                    &[]
                } else {
                    &text1.as_bytes()[offset..]
                }
            },
            None,
            Some(options.reborrow()),
        )
        .unwrap();

    assert_eq!(tree1.root_node().child(0).unwrap().kind(), "function_item");

    let tree2 = parser
        .parse_with_options(
            &mut |offset, _| {
                if offset >= text2.len() {
                    &[]
                } else {
                    &text2.as_bytes()[offset..]
                }
            },
            None,
            Some(options.reborrow()),
        )
        .unwrap();

    assert_eq!(tree2.root_node().child(0).unwrap().kind(), "function_item");

    assert!(parse_count.load(Ordering::SeqCst) > 0);
}

#[test]
fn test_grammar_that_should_hang_and_not_segfault() {
    fn hang_test() {
        let test_grammar_dir = fixtures_dir()
            .join("test_grammars")
            .join("get_col_should_hang_not_crash");

        let grammar_json = load_grammar_file(&test_grammar_dir.join("grammar.js"), None)
            .expect("Failed to load grammar file");

        let (parser_name, parser_code) =
            generate_parser(grammar_json.as_str()).expect("Failed to generate parser");

        let language =
            get_test_language(&parser_name, &parser_code, Some(test_grammar_dir.as_path()));

        let mut parser = Parser::new();
        parser
            .set_language(&language)
            .expect("Failed to set parser language");

        let code_that_should_hang = "\nHello";

        parser
            .parse(code_that_should_hang, None)
            .expect("Parse operation completed unexpectedly");
    }

    let timeout = Duration::from_millis(500);
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || tx.send(std::panic::catch_unwind(hang_test)));

    match rx.recv_timeout(timeout) {
        Ok(Ok(())) => panic!("The test completed rather than hanging"),
        Ok(Err(panic_info)) => panic!("The test panicked unexpectedly: {panic_info:?}"),
        Err(mpsc::RecvTimeoutError::Timeout) => {} // Expected
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            panic!("The test thread disconnected unexpectedly")
        }
    }
}

/// An ambiguous, nested grammar with comment extras, built to drive the
/// error-recovery machinery in parser.c down many different paths. `sum` and
/// `product` are token-identical (a declared conflict), so multiple stack
/// versions compete during both normal parsing and recovery.
fn recovery_torture_language() -> Language {
    let (name, code) = generate_parser(
        r##"{
            "name": "recovery_torture",
            "extras": [
                {"type": "PATTERN", "value": "\\s"},
                {"type": "SYMBOL", "name": "comment"}
            ],
            "conflicts": [["sum", "product"]],
            "rules": {
                "program": {"type": "REPEAT", "content": {"type": "SYMBOL", "name": "statement"}},
                "statement": {"type": "CHOICE", "members": [
                    {"type": "SYMBOL", "name": "block"},
                    {"type": "SYMBOL", "name": "sum"},
                    {"type": "SYMBOL", "name": "product"}
                ]},
                "block": {"type": "SEQ", "members": [
                    {"type": "STRING", "value": "{"},
                    {"type": "REPEAT", "content": {"type": "SYMBOL", "name": "statement"}},
                    {"type": "STRING", "value": "}"}
                ]},
                "sum": {"type": "SEQ", "members": [
                    {"type": "SYMBOL", "name": "id"},
                    {"type": "STRING", "value": "+"},
                    {"type": "SYMBOL", "name": "id"},
                    {"type": "STRING", "value": ";"}
                ]},
                "product": {"type": "SEQ", "members": [
                    {"type": "SYMBOL", "name": "id"},
                    {"type": "STRING", "value": "+"},
                    {"type": "SYMBOL", "name": "id"},
                    {"type": "STRING", "value": ";"}
                ]},
                "comment": {"type": "PATTERN", "value": "#[^\\n]*"},
                "id": {"type": "PATTERN", "value": "[a-z]+"}
            }
        }"##,
    )
    .unwrap();
    get_test_language(&name, &code, None)
}

#[test]
fn test_recovery_torture_log_snapshots() {
    // Full log-stream snapshots for the ambiguous recovery grammar. These pin
    // the error-recovery and tree-selection LOG calls in parser.c that the
    // simple snapshots above do not reach: recover_to_previous,
    // recover_with_missing, select_earlier, select_smaller_error, condense, etc.
    // To update after an intentional change, copy the `left` value from the
    // assertion failure.
    let lang = recovery_torture_language();

    assert_eq!(
        collect_parse_log(&lang, "{ ? ? ? a + b ; }"),
        "\
parse new_parse
parse process version:0, version_count:1, state:1, row:0, col:0
parse lex_internal state:0, row:0, column:0
lex consume character:'{'
parse lexed_lookahead sym:{, size:1
parse shift state:3
parse process version:0, version_count:1, state:3, row:0, col:1
parse lex_internal state:0, row:0, column:1
lex skip character:' '
parse lex_internal state:0, row:0, column:1
lex skip character:' '
parse skip_unrecognized_character
lex consume character:'?'
parse lex_internal state:0, row:0, column:3
lex skip character:' '
parse lex_internal state:0, row:0, column:4
lex consume character:'?'
parse lex_internal state:0, row:0, column:5
lex skip character:' '
parse lex_internal state:0, row:0, column:6
lex consume character:'?'
parse lex_internal state:0, row:0, column:7
lex skip character:' '
lex consume character:'a'
parse lexed_lookahead sym:ERROR, size:6
parse detect_error lookahead:ERROR
parse resume version:0
parse skip_token symbol:ERROR
parse process version:0, version_count:1, state:0, row:0, col:7
parse lex_internal state:0, row:0, column:7
lex skip character:' '
lex consume character:'a'
parse lexed_lookahead sym:id, size:2
parse recover_to_previous state:3, depth:2
parse skip_token symbol:id
parse process version:1, version_count:2, state:3, row:0, col:7
parse shift state:10
parse condense
parse process version:0, version_count:1, state:10, row:0, col:9
parse lex_internal state:0, row:0, column:9
lex skip character:' '
lex consume character:'+'
parse lexed_lookahead sym:+, size:2
parse shift state:12
parse process version:0, version_count:1, state:12, row:0, col:11
parse lex_internal state:0, row:0, column:11
lex skip character:' '
lex consume character:'b'
parse lexed_lookahead sym:id, size:2
parse shift state:13
parse process version:0, version_count:1, state:13, row:0, col:13
parse lex_internal state:0, row:0, column:13
lex skip character:' '
lex consume character:';'
parse lexed_lookahead sym:;, size:2
parse shift state:9
parse process version:0, version_count:1, state:9, row:0, col:15
parse lex_internal state:0, row:0, column:15
lex skip character:' '
lex consume character:'}'
parse lexed_lookahead sym:}, size:2
parse reduce sym:sum, child_count:4
parse reduce sym:product, child_count:4
parse reduce sym:statement, child_count:1
parse select_earlier symbol:statement, over_symbol:statement
parse shift state:8
parse process version:0, version_count:1, state:8, row:0, col:17
parse lex_internal state:0, row:0, column:17
parse lexed_lookahead sym:end, size:0
parse reduce sym:block, child_count:3
parse reduce sym:statement, child_count:1
parse reduce sym:program, child_count:1
parse accept
parse done",
    );

    assert_eq!(
        collect_parse_log(&lang, "x + + y ; { z + w ; }"),
        "\
parse new_parse
parse process version:0, version_count:1, state:1, row:0, col:0
parse lex_internal state:0, row:0, column:0
lex consume character:'x'
parse lexed_lookahead sym:id, size:1
parse shift state:10
parse process version:0, version_count:1, state:10, row:0, col:1
parse lex_internal state:0, row:0, column:1
lex skip character:' '
lex consume character:'+'
parse lexed_lookahead sym:+, size:2
parse shift state:12
parse process version:0, version_count:1, state:12, row:0, col:3
parse lex_internal state:0, row:0, column:3
lex skip character:' '
lex consume character:'+'
parse lexed_lookahead sym:+, size:2
parse detect_error lookahead:+
parse resume version:0
parse recover_to_previous state:10, depth:2
parse skip_token symbol:+
parse process version:0, version_count:2, state:0, row:0, col:5
parse lex_internal state:0, row:0, column:5
lex skip character:' '
lex consume character:'y'
parse lexed_lookahead sym:id, size:2
parse recover_to_previous state:12, depth:2
parse skip_token symbol:id
parse process version:1, version_count:3, state:10, row:0, col:3
parse lex_internal state:0, row:0, column:3
lex skip character:' '
lex consume character:'+'
parse lexed_lookahead sym:+, size:2
parse shift state:12
parse process version:1, version_count:3, state:12, row:0, col:5
parse lex_internal state:0, row:0, column:5
lex skip character:' '
lex consume character:'y'
parse lexed_lookahead sym:id, size:2
parse shift state:13
parse process version:2, version_count:3, state:12, row:0, col:5
parse shift state:13
parse condense
parse process version:0, version_count:1, state:13, row:0, col:7
parse lex_internal state:0, row:0, column:7
lex skip character:' '
lex consume character:';'
parse lexed_lookahead sym:;, size:2
parse shift state:9
parse process version:0, version_count:1, state:9, row:0, col:9
parse lex_internal state:0, row:0, column:9
lex skip character:' '
lex consume character:'{'
parse lexed_lookahead sym:{, size:2
parse reduce sym:sum, child_count:4
parse reduce sym:product, child_count:4
parse reduce sym:statement, child_count:1
parse shift state:3
parse process version:0, version_count:1, state:3, row:0, col:11
parse lex_internal state:0, row:0, column:11
lex skip character:' '
lex consume character:'z'
parse lexed_lookahead sym:id, size:2
parse shift state:10
parse process version:0, version_count:1, state:10, row:0, col:13
parse lex_internal state:0, row:0, column:13
lex skip character:' '
lex consume character:'+'
parse lexed_lookahead sym:+, size:2
parse shift state:12
parse process version:0, version_count:1, state:12, row:0, col:15
parse lex_internal state:0, row:0, column:15
lex skip character:' '
lex consume character:'w'
parse lexed_lookahead sym:id, size:2
parse shift state:13
parse process version:0, version_count:1, state:13, row:0, col:17
parse lex_internal state:0, row:0, column:17
lex skip character:' '
lex consume character:';'
parse lexed_lookahead sym:;, size:2
parse shift state:9
parse process version:0, version_count:1, state:9, row:0, col:19
parse lex_internal state:0, row:0, column:19
lex skip character:' '
lex consume character:'}'
parse lexed_lookahead sym:}, size:2
parse reduce sym:sum, child_count:4
parse reduce sym:product, child_count:4
parse reduce sym:statement, child_count:1
parse select_earlier symbol:statement, over_symbol:statement
parse shift state:8
parse process version:0, version_count:1, state:8, row:0, col:21
parse lex_internal state:0, row:0, column:21
parse lexed_lookahead sym:end, size:0
parse reduce sym:block, child_count:3
parse reduce sym:statement, child_count:1
parse reduce sym:program_repeat1, child_count:2
parse reduce sym:program, child_count:1
parse accept
parse done",
    );

    assert_eq!(
        collect_parse_log(&lang, "{ { { a + b ;"),
        "\
parse new_parse
parse process version:0, version_count:1, state:1, row:0, col:0
parse lex_internal state:0, row:0, column:0
lex consume character:'{'
parse lexed_lookahead sym:{, size:1
parse shift state:3
parse process version:0, version_count:1, state:3, row:0, col:1
parse lex_internal state:0, row:0, column:1
lex skip character:' '
lex consume character:'{'
parse lexed_lookahead sym:{, size:2
parse shift state:3
parse process version:0, version_count:1, state:3, row:0, col:3
parse lex_internal state:0, row:0, column:3
lex skip character:' '
lex consume character:'{'
parse lexed_lookahead sym:{, size:2
parse shift state:3
parse process version:0, version_count:1, state:3, row:0, col:5
parse lex_internal state:0, row:0, column:5
lex skip character:' '
lex consume character:'a'
parse lexed_lookahead sym:id, size:2
parse shift state:10
parse process version:0, version_count:1, state:10, row:0, col:7
parse lex_internal state:0, row:0, column:7
lex skip character:' '
lex consume character:'+'
parse lexed_lookahead sym:+, size:2
parse shift state:12
parse process version:0, version_count:1, state:12, row:0, col:9
parse lex_internal state:0, row:0, column:9
lex skip character:' '
lex consume character:'b'
parse lexed_lookahead sym:id, size:2
parse shift state:13
parse process version:0, version_count:1, state:13, row:0, col:11
parse lex_internal state:0, row:0, column:11
lex skip character:' '
lex consume character:';'
parse lexed_lookahead sym:;, size:2
parse shift state:9
parse process version:0, version_count:1, state:9, row:0, col:13
parse lex_internal state:0, row:0, column:13
parse lexed_lookahead sym:end, size:0
parse reduce sym:sum, child_count:4
parse reduce sym:product, child_count:4
parse reduce sym:statement, child_count:1
parse select_earlier symbol:statement, over_symbol:statement
parse detect_error lookahead:end
parse resume version:0
parse recover_with_missing symbol:}, state:5
parse recover_to_previous state:1, depth:5
parse recover_eof
parse process version:1, version_count:3, state:5, row:0, col:13
parse lex_internal state:0, row:0, column:13
parse lexed_lookahead sym:end, size:0
parse detect_error lookahead:end
parse process version:2, version_count:3, state:1, row:0, col:13
parse reduce sym:program, child_count:0
parse accept
parse resume version:0
parse recover_with_missing symbol:}, state:5
parse recover_eof
parse select_smaller_error symbol:program, over_symbol:ERROR
parse done",
    );

    assert_eq!(
        collect_parse_log(&lang, "? ? ? ?"),
        "\
parse new_parse
parse process version:0, version_count:1, state:1, row:0, col:0
parse lex_internal state:0, row:0, column:0
parse lex_internal state:0, row:0, column:0
parse skip_unrecognized_character
lex consume character:'?'
parse lex_internal state:0, row:0, column:1
lex skip character:' '
parse lex_internal state:0, row:0, column:2
lex consume character:'?'
parse lex_internal state:0, row:0, column:3
lex skip character:' '
parse lex_internal state:0, row:0, column:4
lex consume character:'?'
parse lex_internal state:0, row:0, column:5
lex skip character:' '
parse lex_internal state:0, row:0, column:6
lex consume character:'?'
parse lex_internal state:0, row:0, column:7
parse lexed_lookahead sym:ERROR, size:7
parse detect_error lookahead:ERROR
parse resume version:0
parse skip_token symbol:ERROR
parse process version:0, version_count:1, state:0, row:0, col:7
parse lex_internal state:0, row:0, column:7
parse lexed_lookahead sym:end, size:0
parse recover_to_previous state:1, depth:2
parse recover_eof
parse process version:1, version_count:2, state:1, row:0, col:7
parse reduce sym:program, child_count:0
parse accept
parse done",
    );
}

#[test]
fn test_error_recovery_paths_with_ambiguous_grammar() {
    let language = recovery_torture_language();
    let mut parser = Parser::new();
    parser.set_language(&language).unwrap();
    let mut messages = Vec::new();
    // SAFETY: the logger borrows `messages` and is only invoked during the
    // `parse` call(s) below while `messages` is in scope.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|_, m| messages.push(m.to_string()))));
    }

    // Each input is designed to exercise a different recovery path. They all
    // must parse (with errors) without panicking, leaking, or hanging.
    let cases = [
        // Skip several unrecognized tokens, then re-synchronise to a valid
        // statement inside the block (strategy 1: recover_to_previous).
        "{ ? ? ? a + b ; }",
        // A comment (an `extra`) appears in the middle of an error region, so
        // the token being skipped during recovery is marked extra.
        "a + b ? # trailing comment\n c + d ;",
        // Deeply nested, truncated at EOF: recover_to_state at several depths
        // followed by recover_eof.
        "{ { { a + b ;",
        // Pure garbage to EOF.
        "? ? ? ?",
        // Too many closers: recovery after the statements are already reduced.
        "a + b ; } } }",
        // Error in the middle of the ambiguous rule, forcing competing versions
        // to recover differently.
        "x + + y ; { z + w ; }",
        // Re-sync through an already-built error node (recover_to_state error
        // tree splice): error, skip, then a valid closing brace.
        "{ a + ? ? b ; ? }",
        // Several independent error regions in one block, each recovering and
        // re-synchronising to the next valid statement.
        "{ ? a + b ; ? c + d ; ? e + f ; ? }",
        // Error inside a nested block, then re-sync out to the outer block.
        "{ { ? } a + b ; }",
        // Multi-token skip at the top level, then re-sync to a statement.
        "a + b ; ? ? ? c + d ;",
        // Missing operand, then re-sync to the next statement.
        "{ a + ; b + c ; }",
    ];

    for src in cases {
        let tree = parser.parse(src, None).unwrap();
        assert!(
            tree.root_node().has_error(),
            "expected an error parsing {src:?}"
        );
    }

    // We should have driven both recovery strategies.
    assert!(
        messages.iter().any(|m| m.contains("recover")),
        "expected recovery to be logged"
    );
}

#[test]
fn test_incremental_error_recovery_with_ambiguous_grammar() {
    // Editing erroneous input and reparsing exercises node reuse and the
    // breakdown paths while errors and competing (ambiguous) versions are live.
    let language = recovery_torture_language();
    let mut parser = Parser::new();
    parser.set_language(&language).unwrap();
    let mut messages = Vec::new();
    // SAFETY: the logger borrows `messages` and is only invoked during the
    // `parse` call(s) below while `messages` is in scope.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|_, m| messages.push(m.to_string()))));
    }

    let mut input = b"{ a + b ; c + d ; e + f ; }".to_vec();
    let mut tree = parser.parse(&input, None).unwrap();
    assert!(!tree.root_node().has_error());

    // Break the first statement: reuse of later subtrees must coexist with a new
    // error region at the start.
    perform_edit(
        &mut tree,
        &mut input,
        &Edit {
            position: 4,
            deleted_length: 1,
            inserted_text: b"?".to_vec(),
        },
    )
    .unwrap();
    let mut tree = parser.parse(&input, Some(&tree)).unwrap();
    assert!(tree.root_node().has_error());

    // Now repair it and reparse again, reusing the error region's neighbours.
    let pos = std::str::from_utf8(&input).unwrap().find('?').unwrap();
    perform_edit(
        &mut tree,
        &mut input,
        &Edit {
            position: pos,
            deleted_length: 1,
            inserted_text: b"+".to_vec(),
        },
    )
    .unwrap();
    let tree = parser.parse(&input, Some(&tree)).unwrap();
    assert!(!tree.root_node().has_error());
    assert_ne!(messages.len(), 0);
}

#[test]
fn test_select_tree_by_subtree_comparison() {
    // Two interchangeable rules matching the same input, with equal precedence
    // and zero error cost, force ts_parser__select_tree past the error-cost and
    // precedence checks and into ts_subtree_compare.
    let (name, code) = generate_parser(
        r#"{
            "name": "select_by_compare",
            "extras": [{"type": "PATTERN", "value": "\\s"}],
            "conflicts": [["rule_a", "rule_b"]],
            "rules": {
                "program": {"type": "CHOICE", "members": [
                    {"type": "SYMBOL", "name": "rule_a"},
                    {"type": "SYMBOL", "name": "rule_b"}
                ]},
                "rule_a": {"type": "SEQ", "members": [
                    {"type": "STRING", "value": "x"},
                    {"type": "STRING", "value": "y"}
                ]},
                "rule_b": {"type": "SEQ", "members": [
                    {"type": "STRING", "value": "x"},
                    {"type": "STRING", "value": "y"}
                ]}
            }
        }"#,
    )
    .unwrap();

    let mut parser = Parser::new();
    parser
        .set_language(&get_test_language(&name, &code, None))
        .unwrap();
    let mut messages = Vec::new();
    // SAFETY: the logger borrows `messages` and is only invoked during the
    // `parse` call(s) below while `messages` is in scope.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|_, m| messages.push(m.to_string()))));
    }

    let tree = parser.parse("x y", None).unwrap();
    let kind = tree.root_node().child(0).unwrap().kind();
    assert!(kind == "rule_a" || kind == "rule_b", "got {kind}");
    assert!(
        messages
            .iter()
            .any(|m| m.contains("select_earlier") || m.contains("select_existing")),
        "expected a comparison-based selection to be logged, got: {messages:?}"
    );
}

#[test]
fn test_parser_language_getter() {
    // ts_parser_language: None before a language is set, Some afterwards.
    let mut parser = Parser::new();
    assert!(parser.language().is_none());

    let language = get_language("rust");
    parser.set_language(&language).unwrap();
    let got = parser.language().unwrap();
    assert_eq!(got.node_kind_count(), language.node_kind_count());
    assert_eq!(got.abi_version(), language.abi_version());
}

#[test]
fn test_incremental_parse_with_external_scanner() {
    // Python's indent/dedent external scanner must serialize and restore its
    // state across an incremental reparse (ts_parser__external_scanner_serialize
    // / _deserialize and the reused-node paths).
    let mut parser = Parser::new();
    parser.set_language(&get_language("python")).unwrap();

    let mut input = b"def a():\n    if x:\n        y\n        z\n".to_vec();
    let mut tree = parser.parse(&input, None).unwrap();
    assert!(!tree.root_node().has_error());

    // Insert a new statement inside the nested block, at the same indentation
    // (right before the `z` token).
    let pos = std::str::from_utf8(&input).unwrap().find('z').unwrap();
    perform_edit(
        &mut tree,
        &mut input,
        &Edit {
            position: pos,
            deleted_length: 0,
            inserted_text: b"w\n        ".to_vec(),
        },
    )
    .unwrap();

    let tree2 = parser.parse(&input, Some(&tree)).unwrap();
    assert!(!tree2.root_node().has_error());
    assert!(
        std::str::from_utf8(&input)
            .unwrap()
            .contains("        w\n        z")
    );
}

#[test]
fn test_heavy_error_recovery() {
    // A variety of badly-malformed inputs to drive the error-recovery branches
    // (ts_parser__recover / __handle_error / __recover_to_state).
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();
    for src in [
        "fn fn fn ( ( ( } } ;;;",
        "struct { struct { struct {",
        "let x = = = ;;; fn ) ) (",
        "impl impl for for where where {{{ }}} )))",
        "use use :: :: ;; mod mod {{ pub pub",
        "fn f() -> -> -> { match match { => => } }",
    ] {
        let tree = parser.parse(src, None).unwrap();
        assert!(
            tree.root_node().has_error(),
            "expected an error parsing {src:?}"
        );
    }
}

#[test]
fn test_select_tree_with_ambiguous_grammar() {
    // The `dynamic_precedence` fixture has a runtime conflict between its
    // `expression` and `type`/`declarator` rules: for an ambiguous input, two
    // parse versions both run to completion, so the parser must call
    // `ts_parser__select_tree` to choose between them via dynamic precedence.
    // A logger is attached so select_tree's `LOG()` branches are exercised too.
    let language = get_test_fixture_language("dynamic_precedence");
    let mut parser = Parser::new();
    parser.set_language(&language).unwrap();

    let mut messages = Vec::new();
    // SAFETY: the logger borrows `messages` and is only invoked during the
    // `parse` call(s) below while `messages` is in scope.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|_, message| {
            messages.push(message.to_string());
        })));
    }

    // "T * x" is ambiguous: `* x` is either a declarator (prec.dynamic 1) or a
    // multiplication expression. The dynamic precedence selects the declaration.
    let tree = parser.parse("T * x", None).unwrap();
    assert_eq!(
        tree.root_node().to_sexp(),
        "(program (declaration (type (identifier)) (declarator (identifier))))"
    );
    // The dynamic-precedence comparison branch in select_tree must have fired.
    assert!(
        messages
            .iter()
            .any(|m| m.contains("select_higher_precedence")),
        "expected a dynamic-precedence selection to be logged"
    );

    // An input that only parses as an expression (not a valid declaration).
    let tree = parser.parse("w * x * y", None).unwrap();
    assert_eq!(tree.root_node().child(0).unwrap().kind(), "expression");
}

#[test]
fn test_logging_during_error_recovery_and_incremental_parse() {
    // Parsing with a logger attached executes the bodies of the many `LOG()`
    // statements in parser.c (they are no-ops unless a logger is set). Feeding
    // malformed input and then an incremental edit reaches the LOG branches in
    // the error-recovery and node-reuse paths, not just the happy path.
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let mut messages = Vec::new();
    // SAFETY: the logger borrows `messages` and is only invoked during the
    // `parse` call(s) below while `messages` is in scope.
    unsafe {
        parser.set_logger_unchecked(Some(Box::new(|_, message| {
            messages.push(message.to_string());
        })));
    }

    let mut input = b"fn main() { let x = ; let y = (((; }".to_vec();
    let mut tree = parser.parse(&input, None).unwrap();
    assert!(tree.root_node().has_error());
    assert!(
        messages.iter().any(|m| m.contains("detect_error")),
        "expected error detection to be logged"
    );

    // Incremental reparse after an edit exercises the node-reuse LOG branches.
    let pos = std::str::from_utf8(&input).unwrap().find(';').unwrap();
    perform_edit(
        &mut tree,
        &mut input,
        &Edit {
            position: pos,
            deleted_length: 0,
            inserted_text: b"1".to_vec(),
        },
    )
    .unwrap();
    let tree = parser.parse(&input, Some(&tree)).unwrap();
    assert!(tree.root_node().has_error());
}

#[test]
fn test_incremental_parse_introducing_and_fixing_errors() {
    // Edit a valid tree into an invalid one and back, exercising node reuse
    // interleaved with error recovery across reparses.
    let mut parser = Parser::new();
    parser.set_language(&get_language("rust")).unwrap();

    let mut input = b"fn main() { let x = 1; let y = 2; }".to_vec();
    let mut tree = parser.parse(&input, None).unwrap();
    assert!(!tree.root_node().has_error());

    // Break it: delete a closing brace.
    let pos = std::str::from_utf8(&input).unwrap().rfind('}').unwrap();
    perform_edit(
        &mut tree,
        &mut input,
        &Edit {
            position: pos,
            deleted_length: 1,
            inserted_text: Vec::new(),
        },
    )
    .unwrap();
    let tree = parser.parse(&input, Some(&tree)).unwrap();
    assert!(tree.root_node().has_error());

    // Fix it again.
    let mut tree = tree;
    let end = input.len();
    perform_edit(
        &mut tree,
        &mut input,
        &Edit {
            position: end,
            deleted_length: 0,
            inserted_text: b"}".to_vec(),
        },
    )
    .unwrap();
    let tree = parser.parse(&input, Some(&tree)).unwrap();
    assert!(!tree.root_node().has_error());
}

/// Run `write` (which prints a dot graph to the given file) and return the
/// file's contents.
fn capture_dot_graph(write: impl FnOnce(&std::fs::File)) -> String {
    use std::io::{Read, Seek};
    let mut file = tempfile::tempfile().unwrap();
    write(&file);
    file.rewind().unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    contents
}

/// Dot graphs embed pointer-based node ids (`tree_0x..`, `node_0x..`). Replace
/// each distinct `0x..` address with a stable `0xPTR<n>` token, numbered by
/// first appearance, so the output can be snapshotted deterministically.
fn normalize_dot_graph(s: &str) -> String {
    let mut map: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let (mut i, mut last) = (0usize, 0usize);
    while i + 1 < bytes.len() {
        if bytes[i] == b'0' && bytes[i + 1] == b'x' {
            let mut j = i + 2;
            while j < bytes.len() && (bytes[j] as char).is_ascii_hexdigit() {
                j += 1;
            }
            if j > i + 2 {
                out.push_str(&s[last..i]);
                let tok = &s[i..j];
                let n = map.len();
                let id = *map.entry(tok).or_insert(n);
                out.push_str("0xPTR");
                out.push_str(&id.to_string());
                i = j;
                last = j;
                continue;
            }
        }
        i += 1;
    }
    out.push_str(&s[last..]);
    out
}

// Full dot-graph snapshots. Dot output embeds pointer-based node ids, so each
// distinct address is normalized to a stable 0xPTR<n> token (see
// normalize_dot_graph) before comparison. To update after an intentional
// change, copy the left value from the assertion failure.
#[test]
fn test_tree_dot_graph_snapshot() {
    let mut parser = Parser::new();
    parser.set_language(&recovery_torture_language()).unwrap();
    let tree = parser.parse("a + b ;", None).unwrap();
    let graph = normalize_dot_graph(&capture_dot_graph(|f| tree.print_dot_graph(f)));
    assert_eq!(
        graph.trim_end(),
        r#"digraph tree {
edge [arrowhead=none]
tree_0xPTR0 [label="program", tooltip="range: 0 - 7
state: 0
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 6
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR1 [label="statement", tooltip="range: 0 - 7
state: 65535
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 5
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 [label="sum", tooltip="range: 0 - 7
state: 65535
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 4
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR3 [label="id", shape=plaintext, tooltip="range: 0 - 1
state: 1
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR3 [tooltip=0]
tree_0xPTR4 [label="+", shape=plaintext, tooltip="range: 1 - 3
state: 10
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR4 [tooltip=1]
tree_0xPTR5 [label="id", shape=plaintext, tooltip="range: 3 - 5
state: 12
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR5 [tooltip=2]
tree_0xPTR6 [label=";", shape=plaintext, tooltip="range: 5 - 7
state: 13
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR6 [tooltip=3]
tree_0xPTR1 -> tree_0xPTR2 [tooltip=0]
tree_0xPTR0 -> tree_0xPTR1 [tooltip=0]
tree_0xPTR7 [label="end", shape=plaintext, fontcolor=gray, tooltip="range: 7 - 7
state: 9
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR0 -> tree_0xPTR7 [tooltip=1]
}"#
    );
}

#[test]
fn test_tree_dot_graph_snapshot_with_extra() {
    // The trailing `#c` comment is an `extra`, rendered with fontcolor=gray.
    let mut parser = Parser::new();
    parser.set_language(&recovery_torture_language()).unwrap();
    let tree = parser.parse("a + b ; #c", None).unwrap();
    let graph = normalize_dot_graph(&capture_dot_graph(|f| tree.print_dot_graph(f)));
    assert_eq!(
        graph.trim_end(),
        r#"digraph tree {
edge [arrowhead=none]
tree_0xPTR0 [label="program", tooltip="range: 0 - 10
state: 0
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 7
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR1 [label="statement", tooltip="range: 0 - 7
state: 65535
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 5
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 [label="sum", tooltip="range: 0 - 7
state: 65535
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 4
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR3 [label="id", shape=plaintext, tooltip="range: 0 - 1
state: 1
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR3 [tooltip=0]
tree_0xPTR4 [label="+", shape=plaintext, tooltip="range: 1 - 3
state: 10
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR4 [tooltip=1]
tree_0xPTR5 [label="id", shape=plaintext, tooltip="range: 3 - 5
state: 12
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR5 [tooltip=2]
tree_0xPTR6 [label=";", shape=plaintext, tooltip="range: 5 - 7
state: 13
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR6 [tooltip=3]
tree_0xPTR1 -> tree_0xPTR2 [tooltip=0]
tree_0xPTR0 -> tree_0xPTR1 [tooltip=0]
tree_0xPTR7 [label="comment", shape=plaintext, fontcolor=gray, tooltip="range: 7 - 10
state: 9
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR0 -> tree_0xPTR7 [tooltip=1]
tree_0xPTR8 [label="end", shape=plaintext, fontcolor=gray, tooltip="range: 10 - 10
state: 9
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR0 -> tree_0xPTR8 [tooltip=2]
}"#
    );
}

#[test]
fn test_tree_dot_graph_snapshot_with_error() {
    // `?` is unrecognized, producing ERROR nodes; the leaf records its char.
    let mut parser = Parser::new();
    parser.set_language(&recovery_torture_language()).unwrap();
    let tree = parser.parse("?", None).unwrap();
    assert!(tree.root_node().has_error());
    let graph = normalize_dot_graph(&capture_dot_graph(|f| tree.print_dot_graph(f)));
    assert_eq!(
        graph.trim_end(),
        r#"digraph tree {
edge [arrowhead=none]
tree_0xPTR0 [label="program", tooltip="range: 0 - 1
state: 65535
error-cost: 501
has-changes: 0
depends-on-column: 0
descendant-count: 2
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR1 [label="ERROR", fontcolor=gray, tooltip="range: 0 - 1
state: 0
error-cost: 501
has-changes: 0
depends-on-column: 0
descendant-count: 1
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 [label="_ERROR", tooltip="range: 0 - 1
state: 65535
error-cost: 501
has-changes: 0
depends-on-column: 0
descendant-count: 1
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR3 [label="ERROR", shape=plaintext, tooltip="range: 0 - 1
state: 1
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1
character: '?'"]
tree_0xPTR2 -> tree_0xPTR3 [tooltip=0]
tree_0xPTR1 -> tree_0xPTR2 [tooltip=0]
tree_0xPTR0 -> tree_0xPTR1 [tooltip=0]
tree_0xPTR4 [label="end", shape=plaintext, fontcolor=gray, tooltip="range: 1 - 1
state: 0
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR0 -> tree_0xPTR4 [tooltip=1]
}"#
    );
}

#[test]
fn test_tree_dot_graph_snapshot_with_changes() {
    // After editing (before reparsing), changed nodes are drawn green.
    let mut parser = Parser::new();
    parser.set_language(&recovery_torture_language()).unwrap();
    let mut tree = parser.parse("a + b ;", None).unwrap();
    tree.edit(&InputEdit {
        start_byte: 0,
        old_end_byte: 0,
        new_end_byte: 1,
        start_position: Point::new(0, 0),
        old_end_position: Point::new(0, 0),
        new_end_position: Point::new(0, 1),
    });
    let graph = normalize_dot_graph(&capture_dot_graph(|f| tree.print_dot_graph(f)));
    assert_eq!(
        graph.trim_end(),
        r#"digraph tree {
edge [arrowhead=none]
tree_0xPTR0 [label="program", color=green, penwidth=2, tooltip="range: 0 - 8
state: 0
error-cost: 0
has-changes: 1
depends-on-column: 0
descendant-count: 6
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR1 [label="statement", color=green, penwidth=2, tooltip="range: 0 - 8
state: 65535
error-cost: 0
has-changes: 1
depends-on-column: 0
descendant-count: 5
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 [label="sum", color=green, penwidth=2, tooltip="range: 0 - 8
state: 65535
error-cost: 0
has-changes: 1
depends-on-column: 0
descendant-count: 4
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR3 [label="id", shape=plaintext, color=green, penwidth=2, tooltip="range: 0 - 2
state: 1
error-cost: 0
has-changes: 1
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR3 [tooltip=0]
tree_0xPTR4 [label="+", shape=plaintext, tooltip="range: 2 - 4
state: 10
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR4 [tooltip=1]
tree_0xPTR5 [label="id", shape=plaintext, tooltip="range: 4 - 6
state: 12
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR5 [tooltip=2]
tree_0xPTR6 [label=";", shape=plaintext, tooltip="range: 6 - 8
state: 13
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR2 -> tree_0xPTR6 [tooltip=3]
tree_0xPTR1 -> tree_0xPTR2 [tooltip=0]
tree_0xPTR0 -> tree_0xPTR1 [tooltip=0]
tree_0xPTR7 [label="end", shape=plaintext, fontcolor=gray, tooltip="range: 8 - 8
state: 9
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR0 -> tree_0xPTR7 [tooltip=1]
}"#
    );
}

#[test]
fn test_parser_dot_graphs_snapshot() {
    // Exercises parser.c LOG_STACK/LOG_TREE and ts_stack_print_dot_graph: every
    // parse step emits a label graph, the stack graph, and finally the tree.
    let mut parser = Parser::new();
    parser.set_language(&log_snapshot_language()).unwrap();
    let graph = capture_dot_graph(|f| {
        parser.print_dot_graphs(f);
        parser.parse("ab", None).unwrap();
        parser.stop_printing_dot_graphs();
    });
    assert_eq!(
        normalize_dot_graph(&graph).trim_end(),
        r#"graph {
label="new_parse"
}

graph {
label="process version:0, version_count:1, state:1, row:0, col:0"
}

graph {
label="lex_internal state:0, row:0, column:0"
}

graph {
label="lexed_lookahead sym:a, size:1"
}

graph {
label="shift state:2"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
node_head_0 [shape=none, label=""]
node_head_0 -> node_0xPTR0 [label=0, fontcolor=blue, weight=10000, labeltooltip="node_count: 1
error_cost: 0"]
node_0xPTR0 [label="2" tooltip="position: 1,1
node_count:1
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR0 -> node_0xPTR1 [label="'a'"labeltooltip="error_cost: 0
dynamic_precedence: 0"];
node_0xPTR1 [label="1" tooltip="position: 1,0
node_count:0
error_cost: 0
dynamic_precedence: 0"];
}


graph {
label="process version:0, version_count:1, state:2, row:0, col:1"
}

graph {
label="lex_internal state:0, row:0, column:1"
}

graph {
label="lexed_lookahead sym:b, size:1"
}

graph {
label="shift state:4"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
node_head_0 [shape=none, label=""]
node_head_0 -> node_0xPTR2 [label=0, fontcolor=blue, weight=10000, labeltooltip="node_count: 2
error_cost: 0"]
node_0xPTR2 [label="4" tooltip="position: 1,2
node_count:2
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR2 -> node_0xPTR0 [label="'b'"labeltooltip="error_cost: 0
dynamic_precedence: 0"];
node_0xPTR0 [label="2" tooltip="position: 1,1
node_count:1
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR0 -> node_0xPTR1 [label="'a'"labeltooltip="error_cost: 0
dynamic_precedence: 0"];
node_0xPTR1 [label="1" tooltip="position: 1,0
node_count:0
error_cost: 0
dynamic_precedence: 0"];
}


graph {
label="process version:0, version_count:1, state:4, row:0, col:2"
}

graph {
label="lex_internal state:0, row:0, column:2"
}

graph {
label="lexed_lookahead sym:end, size:0"
}

graph {
label="reduce sym:program, child_count:2"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
node_head_0 [shape=none, label=""]
node_head_0 -> node_0xPTR3 [label=0, fontcolor=blue, weight=10000, labeltooltip="node_count: 3
error_cost: 0"]
node_0xPTR3 [label="3" tooltip="position: 1,2
node_count:3
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR3 -> node_0xPTR1 [label="program"labeltooltip="error_cost: 0
dynamic_precedence: 0"];
node_0xPTR1 [label="1" tooltip="position: 1,0
node_count:0
error_cost: 0
dynamic_precedence: 0"];
}


graph {
label="accept"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
}


graph {
label="done"
}

digraph tree {
edge [arrowhead=none]
tree_0xPTR4 [label="program", tooltip="range: 0 - 2
state: 0
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 2
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR5 [label="a", shape=plaintext, tooltip="range: 0 - 1
state: 1
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR4 -> tree_0xPTR5 [tooltip=0]
tree_0xPTR6 [label="b", shape=plaintext, tooltip="range: 1 - 2
state: 2
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR4 -> tree_0xPTR6 [tooltip=1]
tree_0xPTR7 [label="end", shape=plaintext, fontcolor=gray, tooltip="range: 2 - 2
state: 4
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR4 -> tree_0xPTR7 [tooltip=2]
}"#
    );
}

#[test]
fn test_parser_dot_graphs_snapshot_with_error() {
    // An error parse exercises the recovery branches of ts_stack_print_dot_graph
    // (stack.c): paused heads (color=red), state summaries, and unknown nodes.
    let mut parser = Parser::new();
    parser.set_language(&log_snapshot_language()).unwrap();
    let graph = capture_dot_graph(|f| {
        parser.print_dot_graphs(f);
        parser.parse("ac", None).unwrap();
        parser.stop_printing_dot_graphs();
    });
    assert_eq!(
        normalize_dot_graph(&graph).trim_end(),
        r#"graph {
label="new_parse"
}

graph {
label="process version:0, version_count:1, state:1, row:0, col:0"
}

graph {
label="lex_internal state:0, row:0, column:0"
}

graph {
label="lexed_lookahead sym:a, size:1"
}

graph {
label="shift state:2"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
node_head_0 [shape=none, label=""]
node_head_0 -> node_0xPTR0 [label=0, fontcolor=blue, weight=10000, labeltooltip="node_count: 1
error_cost: 0"]
node_0xPTR0 [label="2" tooltip="position: 1,1
node_count:1
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR0 -> node_0xPTR1 [label="'a'"labeltooltip="error_cost: 0
dynamic_precedence: 0"];
node_0xPTR1 [label="1" tooltip="position: 1,0
node_count:0
error_cost: 0
dynamic_precedence: 0"];
}


graph {
label="process version:0, version_count:1, state:2, row:0, col:1"
}

graph {
label="lex_internal state:0, row:0, column:1"
}

graph {
label="lex_internal state:0, row:0, column:1"
}

graph {
label="skip_unrecognized_character"
}

graph {
label="lex_internal state:0, row:0, column:2"
}

graph {
label="lexed_lookahead sym:ERROR, size:1"
}

graph {
label="detect_error lookahead:ERROR"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
node_head_0 [shape=none, label=""]
node_head_0 -> node_0xPTR0 [color=red label=0, fontcolor=blue, weight=10000, labeltooltip="node_count: 0
error_cost: 500"]
node_0xPTR0 [label="2" tooltip="position: 1,1
node_count:1
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR0 -> node_0xPTR1 [label="'a'"labeltooltip="error_cost: 0
dynamic_precedence: 0"];
node_0xPTR1 [label="1" tooltip="position: 1,0
node_count:0
error_cost: 0
dynamic_precedence: 0"];
}


graph {
label="resume version:0"
}

graph {
label="skip_token symbol:ERROR"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
node_head_0 [shape=none, label=""]
node_head_0 -> node_0xPTR2 [label=0, fontcolor=blue, weight=10000, labeltooltip="node_count: 2
error_cost: 501
summary: 0 2 1"]
node_0xPTR2 [label="?" tooltip="position: 1,2
node_count:3
error_cost: 501
dynamic_precedence: 0"];
node_0xPTR2 -> node_0xPTR3 [label="_ERROR"labeltooltip="error_cost: 501
dynamic_precedence: 0"];
node_0xPTR3 [label="?" tooltip="position: 1,1
node_count:1
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR3 -> node_0xPTR0 [color=red];
node_0xPTR0 [label="2" tooltip="position: 1,1
node_count:1
error_cost: 0
dynamic_precedence: 0"];
node_0xPTR0 -> node_0xPTR1 [label="'a'"labeltooltip="error_cost: 0
dynamic_precedence: 0"];
node_0xPTR1 [label="1" tooltip="position: 1,0
node_count:0
error_cost: 0
dynamic_precedence: 0"];
}


graph {
label="process version:0, version_count:1, state:0, row:0, col:2"
}

graph {
label="lex_internal state:0, row:0, column:2"
}

graph {
label="lexed_lookahead sym:end, size:0"
}

graph {
label="recover_eof"
}

digraph stack {
rankdir="RL";
edge [arrowhead=none]
}


graph {
label="done"
}

digraph tree {
edge [arrowhead=none]
tree_0xPTR4 [label="ERROR", tooltip="range: 0 - 2
state: 0
error-cost: 602
has-changes: 0
depends-on-column: 0
descendant-count: 2
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR5 [label="a", shape=plaintext, tooltip="range: 0 - 1
state: 1
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR4 -> tree_0xPTR5 [tooltip=0]
tree_0xPTR6 [label="_ERROR", tooltip="range: 1 - 2
state: 65535
error-cost: 501
has-changes: 0
depends-on-column: 0
descendant-count: 1
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR7 [label="ERROR", shape=plaintext, tooltip="range: 1 - 2
state: 2
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1
character: 'c'"]
tree_0xPTR6 -> tree_0xPTR7 [tooltip=0]
tree_0xPTR4 -> tree_0xPTR6 [tooltip=1]
tree_0xPTR8 [label="end", shape=plaintext, fontcolor=gray, tooltip="range: 2 - 2
state: 0
error-cost: 0
has-changes: 0
depends-on-column: 0
descendant-count: 0
repeat-depth: 0
lookahead-bytes: 1"]
tree_0xPTR4 -> tree_0xPTR8 [tooltip=2]
}"#
    );
}

/// A grammar with a field, anonymous tokens, and a comment `extra`, used to
/// snapshot node properties.
fn node_props_language() -> Language {
    let (name, code) = generate_parser(
        r##"{
            "name": "node_props",
            "extras": [
                {"type": "PATTERN", "value": "\\s"},
                {"type": "SYMBOL", "name": "comment"}
            ],
            "rules": {
                "program": {"type": "REPEAT", "content": {"type": "SYMBOL", "name": "assignment"}},
                "assignment": {"type": "SEQ", "members": [
                    {"type": "FIELD", "name": "left", "content": {"type": "SYMBOL", "name": "identifier"}},
                    {"type": "STRING", "value": "="},
                    {"type": "FIELD", "name": "right", "content": {"type": "SYMBOL", "name": "identifier"}},
                    {"type": "STRING", "value": ";"}
                ]},
                "comment": {"type": "PATTERN", "value": "#[^\\n]*"},
                "identifier": {"type": "PATTERN", "value": "[a-z]+"}
            }
        }"##,
    )
    .unwrap();
    get_test_language(&name, &code, None)
}

/// Walk the whole tree (including anonymous nodes) and render each node's
/// field, kind, byte range, point range, and flags. Exercises the `node.c`
/// accessors (start/end byte & point, `is_named`/`extra`/`missing`/`error`) together.
fn render_node_properties(tree: &Tree) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let mut cursor = tree.walk();
    let mut depth = 0usize;
    loop {
        let n = cursor.node();
        let field = cursor
            .field_name()
            .map_or(String::new(), |f| format!("{f}: "));
        let sp = n.start_position();
        let ep = n.end_position();
        let _ = writeln!(
            out,
            "{}{}{} bytes={}..{} [{},{}]-[{},{}] named={} extra={} missing={} error={}",
            "  ".repeat(depth),
            field,
            n.kind(),
            n.start_byte(),
            n.end_byte(),
            sp.row,
            sp.column,
            ep.row,
            ep.column,
            n.is_named(),
            n.is_extra(),
            n.is_missing(),
            n.is_error(),
        );
        if cursor.goto_first_child() {
            depth += 1;
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if cursor.goto_parent() {
                depth -= 1;
            } else {
                return out;
            }
        }
    }
}

#[test]
fn test_node_properties_snapshot() {
    // Pins node.c accessors together (byte/point ranges, is_named/extra/...) for a
    // tree with a field, anonymous tokens, and a comment `extra`.
    let mut parser = Parser::new();
    parser.set_language(&node_props_language()).unwrap();
    let tree = parser.parse("x = y ; # c", None).unwrap();
    assert_eq!(
        render_node_properties(&tree).trim_end(),
        r"program bytes=0..11 [0,0]-[0,11] named=true extra=false missing=false error=false
  assignment bytes=0..7 [0,0]-[0,7] named=true extra=false missing=false error=false
    left: identifier bytes=0..1 [0,0]-[0,1] named=true extra=false missing=false error=false
    = bytes=2..3 [0,2]-[0,3] named=false extra=false missing=false error=false
    right: identifier bytes=4..5 [0,4]-[0,5] named=true extra=false missing=false error=false
    ; bytes=6..7 [0,6]-[0,7] named=false extra=false missing=false error=false
  comment bytes=8..11 [0,8]-[0,11] named=true extra=true missing=false error=false"
    );
}

#[test]
fn test_node_properties_snapshot_with_missing_node() {
    // The `;` is inserted as a zero-width MISSING node (is_missing()).
    let mut parser = Parser::new();
    parser.set_language(&node_props_language()).unwrap();
    let tree = parser.parse("x = y", None).unwrap();
    assert_eq!(
        render_node_properties(&tree).trim_end(),
        r"program bytes=0..5 [0,0]-[0,5] named=true extra=false missing=false error=false
  assignment bytes=0..5 [0,0]-[0,5] named=true extra=false missing=false error=false
    left: identifier bytes=0..1 [0,0]-[0,1] named=true extra=false missing=false error=false
    = bytes=2..3 [0,2]-[0,3] named=false extra=false missing=false error=false
    right: identifier bytes=4..5 [0,4]-[0,5] named=true extra=false missing=false error=false
    ; bytes=5..5 [0,5]-[0,5] named=false extra=false missing=true error=false"
    );
}

#[test]
fn test_node_properties_snapshot_with_error_node() {
    // The leading `@` is wrapped in an ERROR node (is_error()).
    let mut parser = Parser::new();
    parser.set_language(&node_props_language()).unwrap();
    let tree = parser.parse("@ x = y ;", None).unwrap();
    assert_eq!(
        render_node_properties(&tree).trim_end(),
        r"program bytes=0..9 [0,0]-[0,9] named=true extra=false missing=false error=false
  ERROR bytes=0..1 [0,0]-[0,1] named=true extra=true missing=false error=true
    ERROR bytes=0..1 [0,0]-[0,1] named=true extra=false missing=false error=true
  assignment bytes=2..9 [0,2]-[0,9] named=true extra=false missing=false error=false
    left: identifier bytes=2..3 [0,2]-[0,3] named=true extra=false missing=false error=false
    = bytes=4..5 [0,4]-[0,5] named=false extra=false missing=false error=false
    right: identifier bytes=6..7 [0,6]-[0,7] named=true extra=false missing=false error=false
    ; bytes=8..9 [0,8]-[0,9] named=false extra=false missing=false error=false"
    );
}

#[test]
fn test_to_sexp_on_anonymous_node() {
    // Calling to_sexp() on an anonymous token node makes it the sexp root, which
    // hits the anonymous-root branch of ts_subtree__write_to_string (subtree.c).
    let mut parser = Parser::new();
    parser.set_language(&node_props_language()).unwrap();
    let tree = parser.parse("x = y ;", None).unwrap();
    let assignment = tree.root_node().child(0).unwrap();
    assert_eq!(assignment.child(0).unwrap().to_sexp(), "(identifier)");
    assert_eq!(assignment.child(1).unwrap().to_sexp(), r#"("=")"#);
    assert_eq!(assignment.child(3).unwrap().to_sexp(), r#"(";")"#);
}

#[test]
fn test_error_recovery_output_snapshots() {
    // Pins the *recovered tree structure* for malformed inputs. A change in
    // error-recovery behaviour (different error wrapping, re-sync point, or
    // missing/unexpected handling) now fails here, instead of silently still
    // producing some erroneous tree. Complements the recovery log snapshots,
    // which pin the recovery *path*; this pins the recovery *output*.
    let lang = recovery_torture_language();
    for (src, expected) in [
        (
            "{ ? ? ? a + b ; }",
            "(program (statement (block (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))))))",
        ),
        (
            "{ { { a + b ;",
            "(program (ERROR (statement (sum (id) (id)))))",
        ),
        ("? ? ? ?", "(program (ERROR (UNEXPECTED '?')))"),
        (
            "a + b ; } } }",
            "(program (statement (sum (id) (id))) (ERROR))",
        ),
        (
            "x + + y ; { z + w ; }",
            "(program (statement (product (id) (ERROR) (id))) (statement (block (statement (sum (id) (id))))))",
        ),
        (
            "{ a + ? ? b ; ? }",
            "(program (statement (block (statement (product (id) (ERROR (UNEXPECTED '?')) (id))) (ERROR (UNEXPECTED '?')))))",
        ),
        (
            "{ { ? } a + b ; }",
            "(program (statement (block (statement (block (ERROR (UNEXPECTED '?')))) (statement (sum (id) (id))))))",
        ),
        (
            "a + b ; ? ? ? c + d ;",
            "(program (statement (sum (id) (id))) (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))))",
        ),
        (
            "{ a + ; b + c ; }",
            "(program (statement (block (statement (product (id) (ERROR (id)) (id))))))",
        ),
    ] {
        let mut parser = Parser::new();
        parser.set_language(&lang).unwrap();
        let tree = parser.parse(src, None).unwrap();
        assert!(
            tree.root_node().has_error(),
            "{src:?} should contain an error"
        );
        assert_eq!(tree.root_node().to_sexp(), expected, "input: {src:?}");
    }
}

#[test]
fn test_recover_with_missing_log_snapshot() {
    // Pins the missing-token recovery strategy in ts_parser__handle_error
    // (the `recover_with_missing` LOG + ts_subtree_new_missing_leaf path), which
    // none of the other log snapshots reach. Input "a + b" is a complete `sum`
    // except for its terminating `;`, so at EOF the parser inserts a MISSING ";"
    // to complete the reduction. This stream also exercises recover_to_previous,
    // recover_eof, and the select_smaller_error tie-break between the competing
    // recovered versions, all in one parse.
    let expected = "\
parse new_parse
parse process version:0, version_count:1, state:1, row:0, col:0
parse lex_internal state:0, row:0, column:0
lex consume character:'a'
parse lexed_lookahead sym:id, size:1
parse shift state:10
parse process version:0, version_count:1, state:10, row:0, col:1
parse lex_internal state:0, row:0, column:1
lex skip character:' '
lex consume character:'+'
parse lexed_lookahead sym:+, size:2
parse shift state:12
parse process version:0, version_count:1, state:12, row:0, col:3
parse lex_internal state:0, row:0, column:3
lex skip character:' '
lex consume character:'b'
parse lexed_lookahead sym:id, size:2
parse shift state:13
parse process version:0, version_count:1, state:13, row:0, col:5
parse lex_internal state:0, row:0, column:5
parse lexed_lookahead sym:end, size:0
parse detect_error lookahead:end
parse resume version:0
parse recover_with_missing symbol:;, state:9
parse recover_to_previous state:1, depth:4
parse recover_eof
parse process version:1, version_count:4, state:9, row:0, col:5
parse reduce sym:sum, child_count:4
parse reduce sym:product, child_count:4
parse reduce sym:statement, child_count:1
parse process version:2, version_count:4, state:4, row:0, col:5
parse reduce sym:program, child_count:1
parse accept
parse select_smaller_error symbol:program, over_symbol:ERROR
parse process version:3, version_count:4, state:1, row:0, col:5
parse reduce sym:program, child_count:0
parse accept
parse select_smaller_error symbol:program, over_symbol:program
parse done";
    assert_eq!(
        collect_parse_log(&recovery_torture_language(), "a + b"),
        expected
    );
}

#[test]
fn test_recover_with_missing_output_snapshots() {
    // Pins the *trees* produced by missing-token recovery: a MISSING leaf is
    // synthesised so an otherwise-complete production can reduce. Complements the
    // log snapshot above (which pins the recovery path) by pinning the recovered
    // output structure and the position the MISSING node is inserted.
    let lang = recovery_torture_language();
    for (src, expected) in [
        (
            "a + b",
            "(program (statement (sum (id) (id) (MISSING \";\"))))",
        ),
        (
            "{ a + b }",
            "(program (statement (block (statement (product (id) (id) (MISSING \";\"))))))",
        ),
        (
            "{ a + b ; c + d }",
            "(program (statement (block (statement (sum (id) (id))) (statement (product (id) (id) (MISSING \";\"))))))",
        ),
        (
            "{ { a + b } }",
            "(program (statement (block (statement (block (statement (product (id) (id) (MISSING \";\"))))))))",
        ),
        (
            "{ a + b c + d ; }",
            "(program (statement (block (statement (product (id) (id) (MISSING \";\"))) (statement (sum (id) (id))))))",
        ),
    ] {
        let mut parser = Parser::new();
        parser.set_language(&lang).unwrap();
        let tree = parser.parse(src, None).unwrap();
        let root = tree.root_node();
        assert!(root.has_error(), "{src:?} should contain an error");
        assert_eq!(root.to_sexp(), expected, "input: {src:?}");
    }
}

#[test]
fn test_error_recovery_tree_shape_snapshots() {
    // Pins the recovered Tree for inputs that exercise recovery paths the other
    // output snapshots don't reach: extras (comments) landing inside an error
    // region, several independent error regions in one container, a stray closer
    // before any valid statement, and the recover_eof case where the ERROR node
    // becomes the tree root itself (no `program` wrapper). These complement the
    // log snapshots by pinning the recovery *output* for each path.
    let lang = recovery_torture_language();
    for (src, expected) in [
        // A comment (extra) interleaved with two separate error regions.
        (
            "a + b ? # trailing comment\n c + d ;",
            "(program (statement (product (id) (ERROR (id) (UNEXPECTED '?')) (comment) (id) (ERROR (id)))))",
        ),
        // Four independent error regions, each re-syncing to the next statement.
        (
            "{ ? a + b ; ? c + d ; ? e + f ; ? }",
            "(program (statement (block (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))) (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))) (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))) (ERROR (UNEXPECTED '?')))))",
        ),
        // A comment immediately before an error region inside a block.
        (
            "{ # note\n ? a + b ; }",
            "(program (statement (block (comment) (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))))))",
        ),
        // A comment landing between the error token and the recovery point.
        (
            "a + ? # x\n b ;",
            "(program (statement (product (id) (ERROR (UNEXPECTED '?')) (comment) (id))))",
        ),
        // An unexpected `{` mid-expression is wrapped in an empty ERROR.
        (
            "a + { b ;",
            "(program (statement (product (id) (ERROR) (id))))",
        ),
        // A stray closer before any statement: leading ERROR sibling.
        ("} a + b ;", "(program (ERROR) (statement (sum (id) (id))))"),
        // A comment followed by garbage at the top level.
        ("# c\n ? ?", "(program (comment) (ERROR (UNEXPECTED '?')))"),
        // Deeply nested + truncated at EOF: recover_eof makes ERROR the root.
        ("{ { { { a + b ;", "(ERROR (statement (sum (id) (id))))"),
        // Alternating garbage and valid statements at the top level.
        (
            "? a + b ; ? c + d ;",
            "(program (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))) (ERROR (UNEXPECTED '?')) (statement (sum (id) (id))))",
        ),
        // A missing operator between two ids: both ids wrapped in one ERROR.
        ("a b ;", "(program (ERROR (id) (id)))"),
        // A doubled operator: the stray `+` becomes an empty ERROR operand.
        (
            "a + + b ;",
            "(program (statement (product (id) (ERROR) (id))))",
        ),
    ] {
        let mut parser = Parser::new();
        parser.set_language(&lang).unwrap();
        let tree = parser.parse(src, None).unwrap();
        let root = tree.root_node();
        assert!(root.has_error(), "{src:?} should contain an error");
        assert_eq!(root.to_sexp(), expected, "input: {src:?}");
    }
}

#[test]
fn test_error_recovery_node_properties() {
    // Pins node-level facts about recovered trees that to_sexp does not capture:
    // the flags (is_error/is_missing/is_extra) and byte ranges of the synthesised
    // ERROR and MISSING nodes, and how has_error propagates up the tree.
    let lang = recovery_torture_language();

    // --- Missing-token recovery: "a + b" inserts a zero-width MISSING ";". ---
    let mut parser = Parser::new();
    parser.set_language(&lang).unwrap();
    let tree = parser.parse("a + b", None).unwrap();
    let root = tree.root_node();
    let sum = root.child(0).unwrap().child(0).unwrap();
    assert_eq!(sum.kind(), "sum");
    let missing = sum.child(3).unwrap();
    assert_eq!(missing.kind(), ";");
    assert!(missing.is_missing(), "the inserted ; must be missing");
    assert!(!missing.is_named());
    assert!(!missing.is_error());
    assert!(!missing.is_extra());
    // A missing node is zero-width, positioned where the real token belonged.
    assert_eq!(missing.start_byte(), 5);
    assert_eq!(missing.end_byte(), 5);
    assert!(missing.has_error(), "a missing node reports has_error");
    // has_error propagates up; the real leaves carry no error.
    assert!(sum.has_error() && root.has_error());
    assert!(!sum.child(0).unwrap().has_error());

    // --- Skipped-token recovery: "a b ;" wraps the run in one ERROR node. ---
    let mut parser = Parser::new();
    parser.set_language(&lang).unwrap();
    let tree = parser.parse("a b ;", None).unwrap();
    let root = tree.root_node();
    let error = root.child(0).unwrap();
    assert_eq!(error.kind(), "ERROR");
    assert!(error.is_error());
    assert!(error.is_named());
    assert!(!error.is_missing());
    // Recovery error nodes are marked extra so they don't perturb field/child
    // indexing of their parent.
    assert!(error.is_extra(), "a recovery ERROR node is extra");
    // The ERROR spans the skipped, non-zero-width region.
    assert_eq!(error.start_byte(), 0);
    assert_eq!(error.end_byte(), 5);
    assert!(error.start_byte() < error.end_byte());
    assert!(error.has_error());
}

// ---------------------------------------------------------------------------
// Coverage-targeted tests for parser.c paths not otherwise exercised. Each
// names the parser.c behaviour it is intended to reach.
// ---------------------------------------------------------------------------

#[test]
fn test_cov_recover_with_extra_token_as_lookahead() {
    // ts_parser__recover marks the lookahead as extra when, mid-recovery, the
    // next token is itself an `extra` (here a comment): the comment is lexed
    // while the parser is still in the error state.
    let lang = recovery_torture_language();
    for src in [
        "a + b ; ? # c\n",
        "a + b ; ? ? # c\n d + e ;",
        "{ a + b ; ? # comment\n }",
        "? ? # c\n a + b ;",
        "a + b ? ? # c\n ; c + d ;",
        "{ a + ? ? # c\n b ; }",
        "a + b ; ? ? ? # c\n ? ? c + d ;",
        "{ ? ? # c\n ? a + b ; }",
    ] {
        let mut parser = Parser::new();
        parser.set_language(&lang).unwrap();
        let tree = parser.parse(src, None).unwrap();
        assert!(tree.root_node().has_error(), "{src:?}");
    }
}

#[test]
fn test_cov_breakdown_top_of_stack_with_error_child() {
    // ts_parser__breakdown_top_of_stack re-pushes the children of a reused
    // pending node; when a child is an ERROR subtree it forces ERROR_STATE.
    // Reached by reparsing edited input whose old tree contains an error region.
    let lang = recovery_torture_language();
    // Several edit positions relative to an error region inside a block, to
    // provoke reuse of an error-containing pending node and its breakdown.
    for (start, pos, ins) in [
        ("{ a + b ; ? ? ? c + d ; e + f ; g + h ; }", 1usize, " "),
        ("{ a + b ; ? ? ? c + d ; e + f ; g + h ; }", 39, " "),
        ("{ a + ? b ; c + d ; e + f ; }", 27, " "),
        ("a + ? b ; c + d ; e + f ; g + h ;", 32, " "),
        ("{ ? a + b ; c + d ; } e + f ;", 28, " "),
    ] {
        let mut parser = Parser::new();
        parser.set_language(&lang).unwrap();
        let mut input = start.as_bytes().to_vec();
        let mut tree = parser.parse(&input, None).unwrap();
        assert!(tree.root_node().has_error());
        perform_edit(
            &mut tree,
            &mut input,
            &Edit {
                position: pos,
                deleted_length: 0,
                inserted_text: ins.as_bytes().to_vec(),
            },
        )
        .unwrap();
        let tree2 = parser.parse(&input, Some(&tree)).unwrap();
        assert!(tree2.root_node().has_error());
    }
}

#[test]
fn test_cov_large_repetition_triggers_balancing_compression() {
    // A very long repetition produces a deep, unbalanced repeat subtree, so
    // ts_parser__balance_subtree runs many compression steps -- including the
    // scaled progress-check interval (operations = i >> 4) for large i.
    let lang = recovery_torture_language();
    let mut parser = Parser::new();
    parser.set_language(&lang).unwrap();
    let src = "a + b ; ".repeat(5000);
    let tree = parser.parse(&src, None).unwrap();
    assert!(!tree.root_node().has_error());
    assert_eq!(tree.root_node().child_count(), 5000);
}

#[test]
fn test_cov_delete_parser_with_pending_old_tree() {
    // When an incremental parse is cancelled, self->old_tree stays assigned;
    // dropping the parser then releases it (the `if (self->old_tree.ptr)` arm
    // of ts_parser_delete).
    let lang = recovery_torture_language();
    let mut parser = Parser::new();
    parser.set_language(&lang).unwrap();

    let mut input = b"a + b ; ".repeat(200);
    let mut tree = parser.parse(&input, None).unwrap();
    perform_edit(
        &mut tree,
        &mut input,
        &Edit {
            position: 0,
            deleted_length: 0,
            inserted_text: b"c + d ; ".to_vec(),
        },
    )
    .unwrap();

    // Cancel the reparse on the very first progress callback, leaving old_tree set.
    let mut fired = false;
    let cancelled = parser.parse_with_options(
        &mut |i, _| {
            let s: &[u8] = if i < input.len() { &input[i..] } else { b"" };
            s
        },
        Some(&tree),
        Some(ParseOptions::new().progress_callback(&mut |_state| {
            fired = true;
            ControlFlow::Break(())
        })),
    );
    assert!(cancelled.is_none());
    assert!(fired);
    drop(parser); // ts_parser_delete with old_tree still pending
}

#[test]
fn test_cov_recovery_actions_with_accept() {
    // ts_parser__process_candidate_recovery_actions hits its `default` arm when
    // a candidate state's action set includes an Accept action -- reached when
    // recovery considers the start state at end of input.
    let lang = recovery_torture_language();
    for src in ["? ? ?", "a + b ; ?", "{ ? ", "? a + b"] {
        let mut parser = Parser::new();
        parser.set_language(&lang).unwrap();
        let tree = parser.parse(src, None).unwrap();
        assert!(tree.root_node().has_error(), "{src:?}");
    }
}

#[test]
fn test_cov_log_lookahead_escapes_control_chars_in_symbol_names() {
    // The LOG_LOOKAHEAD macro escapes control characters in a token's name when
    // a logger is attached. Anonymous string tokens whose literal *is* a control
    // character (\t \n \v \f \r \\) drive every case of that escape switch.
    let (name, code) = generate_parser(
        r#"{
            "name": "escape_tokens",
            "extras": [],
            "rules": {
                "program": {"type": "SEQ", "members": [
                    {"type": "STRING", "value": "\t"},
                    {"type": "STRING", "value": "\n"},
                    {"type": "STRING", "value": "\u000b"},
                    {"type": "STRING", "value": "\u000c"},
                    {"type": "STRING", "value": "\r"},
                    {"type": "STRING", "value": "\\"}
                ]}
            }
        }"#,
    )
    .unwrap();
    let lang = get_test_language(&name, &code, None);
    let log = collect_parse_log(&lang, "\t\n\u{0b}\u{0c}\r\\");
    for esc in [
        r"sym:\t", r"sym:\n", r"sym:\v", r"sym:\f", r"sym:\r", r"sym:\\",
    ] {
        assert!(log.contains(esc), "expected {esc:?} in log:\n{log}");
    }
}

#[test]
fn test_cov_c_string_input_api() {
    // Exercises the C-only convenience entry points ts_parser_parse_string /
    // ts_parser_parse_string_encoding and their backing ts_string_input_read
    // callback. The Rust binding's `parse` uses a closure input, so these C
    // functions are otherwise never reached. Calling them directly keeps the
    // public C string-parsing API covered.
    use tree_sitter::ffi;
    let lang = recovery_torture_language();
    // SAFETY: `Language` is #[repr(transparent)] over `*const TSLanguage`.
    let raw_lang =
        unsafe { *std::ptr::from_ref::<Language>(&lang).cast::<*const ffi::TSLanguage>() };
    unsafe {
        let parser = ffi::ts_parser_new();
        assert!(ffi::ts_parser_set_language(parser, raw_lang));

        // ts_parser_parse_string: drives ts_string_input_read for the content
        // (byte < length) and the terminating empty read (byte >= length).
        let src = b"a + b ;";
        let tree = ffi::ts_parser_parse_string(
            parser,
            core::ptr::null(),
            src.as_ptr().cast(),
            src.len() as u32,
        );
        assert!(!tree.is_null());
        assert!(!ffi::ts_node_has_error(ffi::ts_tree_root_node(tree)));
        ffi::ts_tree_delete(tree);

        // ts_parser_parse_string_encoding with explicit UTF8.
        let tree2 = ffi::ts_parser_parse_string_encoding(
            parser,
            core::ptr::null(),
            src.as_ptr().cast(),
            src.len() as u32,
            ffi::TSInputEncodingUTF8,
        );
        assert!(!tree2.is_null());
        ffi::ts_tree_delete(tree2);

        // Empty input: ts_string_input_read returns "" on the very first read.
        let tree3 = ffi::ts_parser_parse_string(parser, core::ptr::null(), b"".as_ptr().cast(), 0);
        assert!(!tree3.is_null());
        ffi::ts_tree_delete(tree3);

        ffi::ts_parser_delete(parser);
    }
    drop(lang); // keep the language (and raw_lang) alive across the calls above
}

#[test]
fn test_cov_error_lex_run_to_eof() {
    // ts_parser__lex error path: when the run of unrecognized characters reaches
    // EOF with no valid token-start (and no whitespace) before it, the error
    // token is finalized at EOF rather than at a recovery boundary.
    let lang = recovery_torture_language();
    for src in ["?", "???", "@@@", "abc???", "a + b???", "{ a + b ;???"] {
        let mut parser = Parser::new();
        parser.set_language(&lang).unwrap();
        let tree = parser.parse(src, None).unwrap();
        assert!(tree.root_node().has_error(), "{src:?}");
    }
}

#[test]
fn test_cov_reduce_aborts_with_too_many_versions() {
    // A maximally ambiguous grammar (expr -> expr expr | "a") forces a single
    // reduce to spawn more stack versions than MAX_VERSION_COUNT +
    // MAX_VERSION_COUNT_OVERFLOW, exercising the "aborting reduce with too many
    // versions" cleanup in ts_parser__reduce.
    let (name, code) = generate_parser(
        r#"{
            "name": "super_ambiguous",
            "extras": [{"type": "PATTERN", "value": "\\s"}],
            "conflicts": [["expr", "expr"]],
            "rules": {
                "program": {"type": "SYMBOL", "name": "expr"},
                "expr": {"type": "CHOICE", "members": [
                    {"type": "SEQ", "members": [
                        {"type": "SYMBOL", "name": "expr"},
                        {"type": "SYMBOL", "name": "expr"}
                    ]},
                    {"type": "STRING", "value": "a"}
                ]}
            }
        }"#,
    )
    .unwrap();
    let lang = get_test_language(&name, &code, None);
    let mut parser = Parser::new();
    parser.set_language(&lang).unwrap();
    // A long run of the ambiguous token explodes the version count during reduce.
    let tree = parser.parse("a ".repeat(40), None).unwrap();
    assert_eq!(tree.root_node().kind(), "program");
}
