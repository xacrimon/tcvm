pub mod kind;
mod lit;
pub mod machinery;
mod rules;
pub mod syntax;

use std::ops::{Deref, DerefMut};

use cstree::NodeCache;
use machinery::{Span, State};
use syntax::SyntaxNode;

use crate::T;

pub fn parse(
    cache: &mut NodeCache<'static>,
    source: &str,
) -> (SyntaxNode, Vec<ariadne::Report<Span>>) {
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

    fn run(mut self) -> (SyntaxNode, Vec<ariadne::Report<Span>>) {
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

    use cstree::NodeCache;
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
    test!(literal, "test-files/literal.lua");
    //test!(metalua, "test-files/metalua.lua");
    //test!(nbody, "test-files/nbody.lua");
    test!(op_prec, "test-files/op_prec.lua");
    test!(primes, "test-files/primes.lua");
}
