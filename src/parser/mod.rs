pub mod kind;
pub(crate) mod lit;
pub mod machinery;
mod rules;
pub mod syntax;

use std::ops::{Deref, DerefMut};

use cstree::build::NodeCache;
use kind::T;
pub use machinery::LineMap;
use machinery::{Span, State};
use syntax::SyntaxNode;

pub struct Parse {
    pub root: SyntaxNode,
    pub lines: LineMap,
    pub reports: Vec<ariadne::Report<'static, Span>>,
}

pub fn parse(cache: &mut NodeCache<'static>, source: &str) -> Parse {
    Parser::new(cache, source).run()
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

    // Tree text offsets are packed (no trivia), so the line map must be
    // consulted rather than counting newlines in the tree text.
    #[test]
    fn line_map_skips_trivia() {
        let mut cache = NodeCache::new();
        let parse = parse(&mut cache, "a = 1\n\n-- c\n  b = 2\n\n");
        assert!(parse.reports.is_empty());
        // Tokens: `a` `=` `1` on line 1 at packed offsets 0..3, then `b` `=`
        // `2` on line 4 at 3..6.
        let lines: Vec<u32> = (0..6).map(|o| parse.lines.line_at(o)).collect();
        assert_eq!(lines, [1, 1, 1, 4, 4, 4]);
        assert_eq!(parse.lines.last_line(), 4);
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
