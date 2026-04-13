use std::{
    convert::TryFrom,
    fmt::{self, Debug, Display},
    mem,
    ops::{self, Index, Not},
};

use cstree::{GreenNode, GreenNodeBuilder, NodeCache};
use logos::Logos;

use super::kind::SyntaxKind;
use crate::T;

pub struct State<'cache, 'source> {
    cache: &'cache mut NodeCache<'static>,
    tokens: Vec<(SyntaxKind, Span)>,
    cursor: usize,
    source: &'source str,
    events: Vec<Event>,
    reports: Vec<ariadne::Report<Span>>,
}

impl<'cache, 'source> State<'cache, 'source> {
    pub fn new(cache: &'cache mut NodeCache<'static>, source: &'source str) -> Self {
        let mut tokens = Vec::new();
        tokens.extend(
            SyntaxKind::lexer(source)
                .spanned()
                .map(|(kind, range)| (kind, Span::from_range(range))),
        );

        tokens.push((T![eof], Span::from_range(0..0)));
        let estimated_events = source.len() / 4;

        State {
            cache,
            tokens,
            cursor: 0,
            source,
            events: Vec::with_capacity(estimated_events),
            reports: Vec::new(),
        }
    }

    pub fn at(&self) -> SyntaxKind {
        self.tokens[self.cursor].0
    }

    pub fn peek(&self) -> Option<SyntaxKind> {
        self.tokens[self.cursor + 1..]
            .iter()
            .find_map(|(t, _)| t.is_trivia().not().then(|| *t))
    }

    fn span(&self) -> Span {
        self.tokens[self.cursor].1
    }

    pub fn start(&mut self, kind: SyntaxKind) -> Marker {
        let pos = self.events.len();
        Marker::new(self, pos, kind)
    }

    fn events(&mut self) -> &mut Vec<Event> {
        &mut self.events
    }

    pub fn expect(&mut self, kind: SyntaxKind) -> bool {
        if self.at() == kind {
            self.bump();
            true
        } else {
            self.report(
                self.new_error()
                    .with_message("unexpected token")
                    .with_label(self.new_label().with_message(format!(
                        "expected token {} but found {}",
                        kind,
                        self.at()
                    )))
                    .finish(),
            );
            false
        }
    }

    pub fn report(&mut self, error: ariadne::Report<Span>) {
        self.reports.push(error);
    }

    pub fn new_error(&self) -> ariadne::ReportBuilder<Span> {
        ariadne::Report::build(ariadne::ReportKind::Error, (), self.span().start() as usize)
    }

    pub fn new_label(&self) -> ariadne::Label<Span> {
        ariadne::Label::new(self.span())
    }

    fn bump(&mut self) {
        self.events.push(Event::Token {
            kind: self.at(),
            span: self.span(),
        });

        self.cursor += 1;
    }

    pub fn source(&self, span: Span) -> &str {
        &self.source[span]
    }

    pub fn error_eat_until(&mut self, one_of: &[SyntaxKind]) -> Span {
        let marker = self.start(T![invalid]);
        let mut last_span = self.span();
        while !one_of.contains(&self.at()) {
            self.bump();
            last_span = self.span();
        }

        marker.complete(self);
        last_span
    }

    pub fn finish(self) -> (GreenNode, Vec<ariadne::Report<Span>>) {
        let tree = Sink::new(self.cache, &self.tokens, self.events, self.source).finish();
        (tree, self.reports)
    }
}

pub const INDEX_BINDING_POWER: i32 = 22;
pub const CALL_BINDING_POWER: i32 = 22;

pub fn prefix_binding_power(op: SyntaxKind) -> ((), i32) {
    match op {
        T![not] | T![+] | T![-] | T![#] | T![~] => ((), 21),
        _ => unreachable!(),
    }
}

pub fn infix_binding_power(op: SyntaxKind) -> Option<(i32, i32)> {
    Some(match op {
        T![or] => (1, 2),
        T![and] => (3, 4),
        T![<] | T![>] | T![<=] | T![>=] | T![~=] | T![==] => (5, 6),
        T![|] => (7, 8),
        T![~] => (9, 10),
        T![&] => (11, 12),
        T![<<] | T![>>] => (13, 14),
        T![..] => (16, 15),
        T![+] | T![-] => (17, 18),
        T![*] | T![/] | T![D/] | T![%] => (19, 20),
        T![^] => (22, 21),
        T![.] | T![:] => (24, 23),
        _ => return None,
    })
}

pub fn token_is_literal(token: SyntaxKind) -> bool {
    matches!(
        token,
        T![nil]
            | T![false]
            | T![true]
            | T![int]
            | T![hex_int]
            | T![float]
            | T![hex_float]
            | T![string]
            | T![long_string]
    )
}

pub fn token_is_expr_start(token: SyntaxKind) -> bool {
    token == T![ident]
        || token == T!['(']
        || token_is_literal(token)
        || token_is_unary_op(token)
        || token == T!['{']
        || token == T![function]
        || token == T![...]
}

pub fn token_is_unary_op(token: SyntaxKind) -> bool {
    matches!(token, T![not] | T![+] | T![-] | T![#] | T![~])
}

pub fn token_is_binary_op(token: SyntaxKind) -> bool {
    matches!(
        token,
        T![or]
            | T![and]
            | T![+]
            | T![-]
            | T![*]
            | T![/]
            | T![D/]
            | T![^]
            | T![%]
            | T![&]
            | T![|]
            | T![<<]
            | T![>>]
            | T![==]
            | T![~]
            | T![~=]
            | T![<=]
            | T![>=]
            | T![>]
            | T![<]
            | T![.]
            | T![..]
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    Enter {
        kind: SyntaxKind,
        preceded_by: usize,
    },
    Exit,
    Token {
        kind: SyntaxKind,
        span: Span,
    },
}

impl Event {
    fn tombstone() -> Self {
        Self::Enter {
            kind: T![tombstone],
            preceded_by: 0,
        }
    }

    fn is_tombstone(self) -> bool {
        matches!(
            self,
            Self::Enter {
                kind: T![tombstone],
                preceded_by: 0,
            },
        )
    }
}

impl Default for Event {
    fn default() -> Self {
        Self::tombstone()
    }
}

pub struct Marker {
    position: usize,
    kind: SyntaxKind,
}

impl Marker {
    pub fn new(state: &mut State, position: usize, kind: SyntaxKind) -> Self {
        state.events().push(Event::Enter {
            kind,
            preceded_by: 0,
        });

        Self { position, kind }
    }

    pub fn complete(self, state: &mut State) -> CompletedMarker {
        state.events().push(Event::Exit);
        CompletedMarker {
            position: self.position,
            kind: self.kind,
        }
    }

    pub fn retype(self, state: &mut State, kind: SyntaxKind) -> Self {
        let event_at_pos = &mut state.events()[self.position];
        debug_assert_eq!(*event_at_pos, Event::tombstone());

        *event_at_pos = Event::Enter {
            kind,
            preceded_by: 0,
        };

        self
    }

    pub fn abandon(self, state: &mut State) {
        match &mut state.events()[self.position] {
            Event::Enter {
                kind,
                preceded_by: 0,
            } => {
                *kind = T![tombstone];
            }

            _ => unreachable!(),
        }

        if self.position == state.events().len() - 1 {
            state.events().pop();
        }
    }
}

#[derive(Debug)]
pub struct CompletedMarker {
    position: usize,
    kind: SyntaxKind,
}

impl CompletedMarker {
    pub fn precede(self, state: &mut State, kind: SyntaxKind) -> Marker {
        let marker = state.start(kind);

        if let Event::Enter { preceded_by, .. } = &mut state.events()[self.position] {
            *preceded_by = marker.position - self.position;
        } else {
            unreachable!();
        }

        marker
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span {
    start: u32,
    end: u32,
}

impl Span {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    pub fn start(self) -> u32 {
        self.start
    }

    pub fn end(self) -> u32 {
        self.end
    }

    fn from_range(range: ops::Range<usize>) -> Self {
        debug_assert_eq!(
            u32::try_from(range.start),
            Ok(range.start as u32),
            "range {} out of 32bit bounds (max is {})",
            range.start,
            u32::MAX,
        );
        debug_assert_eq!(
            u32::try_from(range.end),
            Ok(range.end as u32),
            "range {} out of 32bit bounds (max is {})",
            range.end,
            u32::MAX,
        );

        Self::new(range.start as u32, range.end as u32)
    }

    pub fn range(self) -> ops::Range<usize> {
        self.start() as usize..self.end() as usize
    }
}

impl ariadne::Span for Span {
    type SourceId = ();

    fn source(&self) -> &Self::SourceId {
        &()
    }

    fn start(&self) -> usize {
        self.start as usize
    }

    fn end(&self) -> usize {
        self.end as usize
    }
}

impl Index<Span> for str {
    type Output = str;

    fn index(&self, index: Span) -> &Self::Output {
        let range: ops::Range<usize> = index.range();
        &self[range]
    }
}

impl From<Span> for ops::Range<u32> {
    fn from(range: Span) -> Self {
        range.start()..range.end()
    }
}

impl Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.start, f)?;
        f.write_str("..")?;
        Debug::fmt(&self.end, f)
    }
}

impl Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.start, f)?;
        f.write_str("..")?;
        Display::fmt(&self.end, f)
    }
}

struct Sink<'cache, 'source> {
    builder: GreenNodeBuilder<'cache, 'static>,
    tokens: &'source [(SyntaxKind, Span)],
    cursor: usize,
    events: Vec<Event>,
    source: &'source str,
}

impl<'cache, 'source> Sink<'cache, 'source> {
    fn new(
        cache: &'cache mut NodeCache<'static>,
        tokens: &'source [(SyntaxKind, Span)],
        events: Vec<Event>,
        source: &'source str,
    ) -> Self {
        Self {
            builder: GreenNodeBuilder::with_cache(cache),
            tokens,
            cursor: 0,
            events,
            source,
        }
    }

    fn token(&mut self, kind: SyntaxKind, text: &str) {
        self.cursor += 1;
        self.builder.token(kind.into(), text);
    }

    fn finish(mut self) -> GreenNode {
        let mut preceded_nodes = Vec::new();
        for idx in 0..self.events.len() {
            match mem::take(&mut self.events[idx]) {
                // Ignore tombstone events
                event @ Event::Enter { .. } if event.is_tombstone() => {}

                Event::Enter { kind, preceded_by } => {
                    preceded_nodes.push(kind);

                    let (mut idx, mut preceded_by) = (idx, preceded_by);
                    while preceded_by > 0 {
                        idx += preceded_by;

                        preceded_by = match mem::take(&mut self.events[idx]) {
                            Event::Enter { kind, preceded_by } => {
                                if kind != T![tombstone] {
                                    preceded_nodes.push(kind);
                                }

                                preceded_by
                            }

                            _ => unreachable!(),
                        }
                    }

                    #[allow(clippy::iter_with_drain)]
                    for kind in preceded_nodes.drain(..).rev() {
                        self.builder.start_node(kind.into());
                    }
                }

                Event::Exit => {
                    self.builder.finish_node();
                }

                Event::Token { kind, span } => {
                    self.token(kind, &self.source[span]);
                }
            }
        }

        self.builder.finish().0
    }
}
