pub mod kind;
pub(crate) mod lit;
pub mod machinery;
mod rules;
pub mod syntax;

use std::ops::{Deref, DerefMut};

use cstree::build::NodeCache;
use kind::T;
pub use machinery::LineMap;
use machinery::State;
pub use machinery::SyntaxReport;
use syntax::SyntaxNode;

pub struct Parse {
    pub root: SyntaxNode,
    pub lines: LineMap,
    pub reports: Vec<SyntaxReport>,
}

pub fn parse(cache: &mut NodeCache<'static>, source: &str) -> Parse {
    Parser::new(cache, source).run()
}

/// Render `reports` against `source`, which is labelled `name` (a chunk id
/// such as `[string "..."]`), with ANSI colors when `color`.
pub(crate) fn render_reports(
    reports: &[SyntaxReport],
    source: &str,
    name: &str,
    color: bool,
) -> String {
    struct Chunk<'a> {
        name: &'a str,
        source: ariadne::Source<&'a str>,
    }

    impl<'a> ariadne::Cache<()> for Chunk<'a> {
        type Storage = &'a str;

        fn fetch(&mut self, _: &()) -> Result<&ariadne::Source<&'a str>, impl std::fmt::Debug> {
            Ok::<_, std::convert::Infallible>(&self.source)
        }

        fn display<'b>(&self, _: &'b ()) -> Option<impl std::fmt::Display + 'b> {
            Some(self.name.to_owned())
        }
    }

    let mut chunk = Chunk {
        name,
        source: ariadne::Source::from(source),
    };
    // Byte indices match the lexer's spans; ariadne defaults to chars.
    let config = ariadne::Config::default()
        .with_color(color)
        .with_index_type(ariadne::IndexType::Byte);
    let mut out = Vec::new();
    for r in reports {
        let label = ariadne::Label::new(r.span)
            .with_message(&r.label)
            .with_color(ariadne::Color::Red);
        ariadne::Report::build(ariadne::ReportKind::Error, r.span)
            .with_config(config)
            .with_message(&r.message)
            .with_label(label)
            .finish()
            .write(&mut chunk, &mut out)
            .expect("writing to a Vec cannot fail");
    }
    let mut out = String::from_utf8(out).expect("ariadne writes UTF-8");
    out.truncate(out.trim_end().len());
    out
}

struct Parser<'cache, 'source> {
    state: State<'cache, 'source>,
}

impl<'cache, 'source> Parser<'cache, 'source> {
    fn new(cache: &'cache mut NodeCache<'static>, source: &'source str) -> Self {
        Self {
            state: State::new(cache, source),
        }
    }

    fn root(&mut self) {
        let marker = self.start(T![root]);
        self.r_items();
        marker.complete(self);
    }

    fn run(mut self) -> Parse {
        self.root();
        let (root, lines, reports) = self.state.finish();
        Parse {
            root: SyntaxNode::new_root(root),
            lines,
            reports,
        }
    }
}

impl<'cache, 'source> Deref for Parser<'cache, 'source> {
    type Target = State<'cache, 'source>;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl<'cache, 'source> DerefMut for Parser<'cache, 'source> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use cstree::build::NodeCache;
    use insta::assert_snapshot;
    use paste::paste;

    use super::parse;

    macro_rules! test {
        ($name:ident, $path:literal) => {
            paste! {
                #[test]
                fn [<test_parse_ $name>]() {
                    let mut cache = NodeCache::new();
                    let source = fs::read_to_string($path).unwrap();
                    let parse = parse(&mut cache, &source);
                    let syntax_tree = parse.root.debug(cache.interner(), true);
                    assert!(parse.reports.is_empty());
                    assert_snapshot!(syntax_tree);
                }
            }
        };
    }

    test!(comment, "test-files/comment.lua");
    test!(declare, "test-files/declare.lua");
    test!(function, "test-files/function.lua");
    test!(method_def, "test-files/method_def.lua");
    test!(prefix_attrib, "test-files/prefix_attrib.lua");
    test!(hello, "test-files/hello.lua");
    test!(if, "test-files/if.lua");
    test!(jens, "test-files/jens.lua");
    test!(literal, "test-files/literal.lua");
    test!(hex_numeral, "test-files/hex_numeral.lua");
    test!(decimal_numeral, "test-files/decimal_numeral.lua");
    test!(int_overflow_numeral, "test-files/int_overflow_numeral.lua");
    test!(nbody, "test-files/nbody.lua");
    test!(op_prec, "test-files/op_prec.lua");
    test!(primes, "test-files/primes.lua");
    test!(global_decl, "test-files/global_decl.lua");
    test!(global_star, "test-files/global_star.lua");
    test!(
        global_const_assign_err,
        "test-files/global_const_assign_err.lua"
    );
    test!(
        global_undeclared_err,
        "test-files/global_undeclared_err.lua"
    );
    test!(
        for_counter_readonly_err,
        "test-files/for_counter_readonly_err.lua"
    );
    test!(errnnil_runtime, "test-files/errnnil_runtime.lua");
    test!(
        global_nested_propagation,
        "test-files/global_nested_propagation.lua"
    );
    test!(global_star_nested, "test-files/global_star_nested.lua");
    test!(global_const_star, "test-files/global_const_star.lua");
    test!(vararg_param, "test-files/vararg_param.lua");
    test!(paren_prefix, "test-files/paren_prefix.lua");
    test!(call_sugar, "test-files/call_sugar.lua");

    // A malformed tail with no statement-recovery token before EOF (e.g. the
    // adjacent `Float Float` from `1.2.3` / `10..20`) must yield a parse error
    // report, not run the recovery loop past EOF and panic.
    #[test]
    fn malformed_numeral_reports_without_panic() {
        for src in [
            "1.2.3",
            "x = 10..20",
            "print(.5.5)",
            "y = 0x1.2.3",
            "foo @ bar",
        ] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert!(!reports.is_empty(), "expected a parse error for {src:?}");
        }
    }

    // A bad statement inside a block must stop recovery at the block's `end`
    // (one report, parsing resumes after it) rather than eat to EOF, where
    // the block loop used to spin forever (#205).
    #[test]
    fn bad_statement_in_block_recovers_at_end() {
        for src in [
            "do [ end x = 1",
            "while true do ] end x = 1",
            "function f() 1 end x = 1",
            "if x then + end x = 1",
            "if x then + else y() end x = 1",
            "repeat ] until x x = 1",
            "local f = function() 1 end x = 1",
            "do a[ end x = 1",
            "do ( end x = 1",
            "do a[ local y = 1 end x = 1",
        ] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert_eq!(reports.len(), 1, "expected one parse error for {src:?}");
        }

        for src in ["do", "do x = 1", "end", "x() end y()", "repeat x() end"] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert!(!reports.is_empty(), "expected a parse error for {src:?}");
        }
    }

    // A statement that fails partway still closes its node, so the tree
    // builder doesn't panic on the unbalanced events (#234).
    #[test]
    fn failed_statement_after_another_reports_without_panic() {
        for src in [
            "x = 1 if x then",
            "local a\nif x then",
            "::l::\nif x then",
            "x = 1 if x then else",
            "x = 1 if x then elseif y then",
            "x = 1 function f(1) end",
            "x = 1 local f = function(a b) end",
        ] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert!(!reports.is_empty(), "expected a parse error for {src:?}");
        }
    }

    // A missing expression is reported once, where it is missing, and every
    // node on the way out is still closed (#207).
    #[test]
    fn missing_expression_reports_once() {
        for (src, at) in [
            ("x =", 3),
            ("local x =", 9),
            ("global x =", 10),
            ("x = 1,", 6),
            ("return 1,", 9),
            ("do x = ( end y = 1", 9),
            ("if a[ then x() end y = 1", 6),
            ("print(1 +)", 9),
            ("print(1, 2,)", 11),
            ("f(", 2),
            ("x = (1 +) y = 2", 8),
            ("local x = (1 +) y = 2", 14),
            ("return (1 +)", 11),
            ("for i in do end", 9),
            ("while do end", 6),
            ("x = - -", 7),
            ("x = 1 + * 2", 8),
            ("x = {1, 2 +}", 11),
        ] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert_eq!(reports.len(), 1, "expected one parse error for {src:?}");
            assert_eq!(
                (reports[0].message.as_str(), reports[0].span.start()),
                ("expected an expression", at),
                "{src:?}"
            );
        }
    }

    // `return` must end its block, optionally followed by one `;` (#205).
    #[test]
    fn return_must_be_last_statement() {
        for src in [
            "do return {1}[3] end",
            "do return 1 1 end",
            "do return 1 x = 2 end",
            "do return 1 print(\"after\") end print(\"next\")",
            "do return;; end",
            "do return 1 ; x() end",
            "return 1 return 2",
        ] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert!(!reports.is_empty(), "expected a parse error for {src:?}");
        }

        for src in [
            "return",
            "return 1, 2;",
            "do return end",
            "do return 1; end",
            "if x then return 1 elseif y then return 2 else return 3 end",
            "repeat return until x",
            "local function f() return 1 end",
        ] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert!(reports.is_empty(), "unexpected parse error for {src:?}");
        }
    }

    fn rendered_reports(src: &str) -> Vec<String> {
        let mut cache = NodeCache::new();
        parse(&mut cache, src)
            .reports
            .iter()
            .map(|r| super::render_reports(std::slice::from_ref(r), src, "", false))
            .collect()
    }

    // Reports point at and quote the offending token, not wherever recovery
    // or lookahead stopped.
    #[test]
    fn reports_blame_the_offending_token() {
        for (src, at, message) in [
            ("x = 1 ] y = 2", ":1:7 ", "got \"]\""),
            ("do return;; end", ":1:11 ", "found ;"),
            ("do return 1 ; x() end", ":1:15 ", "found ident"),
            ("x = 1 if x then", ":1:16 ", "found eof"),
        ] {
            let reports = rendered_reports(src);
            assert_eq!(reports.len(), 1, "expected one parse error for {src:?}");
            assert!(
                reports[0].contains(at) && reports[0].contains(message),
                "{src:?} should report {message:?} at {at:?}, got:\n{}",
                reports[0]
            );
        }
    }

    // Tree text offsets are packed (no trivia), so the line map must be
    // consulted rather than counting newlines in the tree text.
    #[test]
    fn line_map_skips_trivia() {
        let mut cache = NodeCache::new();
        let p = parse(&mut cache, "a = 1\n\n-- c\n  b = 2\n\n");
        assert!(p.reports.is_empty());
        // Tokens: `a` `=` `1` on line 1 at packed offsets 0..3, then `b` `=`
        // `2` on line 4 at 3..6.
        let lines: Vec<u32> = (0..6).map(|o| p.lines.line_at(o)).collect();
        assert_eq!(lines, [1, 1, 1, 4, 4, 4]);
        assert_eq!(p.lines.last_line(), 4);

        // Newlines inside a token (a long string here) count towards the
        // tokens after it.
        let p = parse(&mut cache, "x = [[a\nb]]\ny = 1");
        assert!(p.reports.is_empty());
        // `x` `=` `[[a\nb]]` occupy packed offsets 0..9; `y` starts at 9.
        assert_eq!(p.lines.line_at(2), 1);
        assert_eq!(p.lines.line_at(9), 3);
        assert_eq!(p.lines.last_line(), 3);
    }

    // Assignment targets must be variables (#67); a parenthesised primary
    // only takes suffixes, never infix operators (#68).
    #[test]
    fn invalid_assignment_targets_are_rejected() {
        for src in [
            "f() = 1",
            "a:m() = 1",
            "(a) = 1",
            "a.b, f() = 1, 2",
            "(a) + b = 1",
            "(f()) = 1",
            // Suffixes only follow a prefixexp (#150).
            "x = {} (1)",
            "x = 1 + 2 [1]",
            "x = 'a' .. 'b' :upper()",
        ] {
            let mut cache = NodeCache::new();
            let reports = parse(&mut cache, src).reports;
            assert!(!reports.is_empty(), "expected a parse error for {src:?}");
        }
    }
}
