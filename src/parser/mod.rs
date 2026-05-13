pub mod kind;
mod lit;
pub mod machinery;
mod rules;
pub mod syntax;

use std::ops::{Deref, DerefMut};

use cstree::build::NodeCache;
use machinery::{Span, State};
use syntax::SyntaxNode;

use crate::T;

pub fn parse(
    cache: &mut NodeCache<'static>,
    source: &str,
) -> (SyntaxNode, Vec<ariadne::Report<'static, Span>>) {
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

    fn run(mut self) -> (SyntaxNode, Vec<ariadne::Report<'static, Span>>) {
        self.root();
        let (root, reports) = self.state.finish();
        (SyntaxNode::new_root(root), reports)
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
                    let (syntax_tree, reports) = parse(&mut cache, &source);
                    let syntax_tree = syntax_tree.debug(cache.interner(), true);
                    assert!(reports.is_empty());
                    assert_snapshot!(syntax_tree);
                }
            }
        };
    }

    test!(comment, "test-files/comment.lua");
    test!(declare, "test-files/declare.lua");
    test!(function, "test-files/function.lua");
    test!(hello, "test-files/hello.lua");
    test!(if, "test-files/if.lua");
    test!(jens, "test-files/jens.lua");
    test!(literal, "test-files/literal.lua");
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
}
