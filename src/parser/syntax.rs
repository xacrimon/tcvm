use std::mem;

use cstree::{
    RawSyntaxKind, Syntax, interning::TokenInterner, syntax::SyntaxText, util::NodeOrToken,
};

use super::{kind::SyntaxKind, lit};
use crate::T;

impl Syntax for SyntaxKind {
    fn from_raw(raw: RawSyntaxKind) -> Self {
        debug_assert!(raw.0 < mem::variant_count::<SyntaxKind>() as u32);
        unsafe { std::mem::transmute(raw.0) }
    }

    fn into_raw(self) -> RawSyntaxKind {
        RawSyntaxKind(self as u32)
    }

    fn static_text(self) -> Option<&'static str> {
        None
    }
}

pub type SyntaxNode = cstree::syntax::SyntaxNode<SyntaxKind>;
pub type SyntaxToken = cstree::syntax::SyntaxToken<SyntaxKind>;
pub type SyntaxElement = NodeOrToken<SyntaxNode, SyntaxToken>;

macro_rules! ast_node {
    ($name:ident, $kind:expr) => {
        #[derive(PartialEq, Eq, Hash)]
        pub struct $name(SyntaxNode);
        impl $name {
            fn cast(node: &SyntaxNode) -> Option<Self> {
                if node.kind() == $kind {
                    Some(Self(node.clone()))
                } else {
                    None
                }
            }
        }
    };
}

ast_node!(Root, T![root]);

impl Root {
    pub fn new(node: SyntaxNode) -> Option<Self> {
        Self::cast(&node)
    }

    pub fn block(&self) -> impl Iterator<Item = Stmt> + '_ {
        self.0.children().filter_map(Stmt::cast)
    }
}

pub enum Stmt {
    Label(Label),
    Goto(Goto),
    Decl(Decl),
    Global(Global),
    Assign(Assign),
    Func(Func),
    Expr(Expr),
    Break(Break),
    Return(Return),
    Do(Do),
    While(While),
    Repeat(Repeat),
    If(If),
    ForNum(ForNum),
    ForGen(ForGen),
}

impl Stmt {
    fn cast(node: &SyntaxNode) -> Option<Self> {
        Some(match node.kind() {
            T![label] => Label::cast(node).map(Self::Label)?,
            T![goto] => Goto::cast(node).map(Self::Goto)?,
            T![decl_stmt] => Decl::cast(node).map(Self::Decl)?,
            T![global_stmt] => Global::cast(node).map(Self::Global)?,
            T![assign_stmt] => Assign::cast(node).map(Self::Assign)?,
            T![func_stmt] => Func::cast(node).map(Self::Func)?,
            T![break_stmt] => Break::cast(node).map(Self::Break)?,
            T![return_stmt] => Return::cast(node).map(Self::Return)?,
            T![do_stmt] => Do::cast(node).map(Self::Do)?,
            T![while_stmt] => While::cast(node).map(Self::While)?,
            T![repeat_stmt] => Repeat::cast(node).map(Self::Repeat)?,
            T![if_stmt] => If::cast(node).map(Self::If)?,
            T![for_num_stmt] => ForNum::cast(node).map(Self::ForNum)?,
            T![for_gen_stmt] => ForGen::cast(node).map(Self::ForGen)?,
            _ => Expr::cast(node).map(Self::Expr)?,
        })
    }
}

ast_node!(Label, T![label]);

impl Label {
    pub fn name(&self) -> Option<Ident> {
        self.0.first_child().and_then(Ident::cast)
    }
}

ast_node!(Goto, T![goto]);

impl Goto {
    pub fn label(&self) -> Option<Ident> {
        self.0.first_child().and_then(Ident::cast)
    }
}

pub enum Expr {
    Method(MethodCall),
    Ident(Ident),
    Literal(Literal),
    Func(FuncExpr),
    Table(Table),
    PrefixOp(PrefixOp),
    BinaryOp(BinaryOp),
    FuncCall(FuncCall),
    Index(Index),
    VarArg,
}

impl Expr {
    fn cast(node: &SyntaxNode) -> Option<Self> {
        Some(match node.kind() {
            T![method_call] => MethodCall::cast(node).map(Self::Method)?,
            T![ident] => Ident::cast(node).map(Self::Ident)?,
            T![vararg_expr] => Self::VarArg,
            T![func_expr] => FuncExpr::cast(node).map(Self::Func)?,
            T![table_expr] => Table::cast(node).map(Self::Table)?,
            T![prefix_op] => PrefixOp::cast(node).map(Self::PrefixOp)?,
            T![bin_op] => BinaryOp::cast(node).map(Self::BinaryOp)?,
            T![func_call] => FuncCall::cast(node).map(Self::FuncCall)?,
            T![index] => Index::cast(node).map(Self::Index)?,
            T![expr] => node.first_child().and_then(Expr::cast)?,
            T![literal_expr] => Literal::cast(node).map(Self::Literal)?,
            _ => return None,
        })
    }
}

ast_node!(MethodCall, T![method_call]);

impl MethodCall {
    pub fn object(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn method(&self) -> Option<Ident> {
        self.0.children().nth(1).and_then(Ident::cast)
    }

    pub fn args(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        Some(self.0.last_child()?.children().filter_map(Expr::cast))
    }
}

ast_node!(Decl, T![decl_stmt]);

impl Decl {
    pub fn function(&self) -> Option<Func> {
        self.0.first_child().and_then(Func::cast)
    }

    pub fn targets(&self) -> Option<impl Iterator<Item = DeclTarget> + '_> {
        Some(
            self.0
                .first_child()?
                .children()
                .filter_map(DeclTarget::cast),
        )
    }

    pub fn values(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        Some(self.0.last_child()?.children().filter_map(Expr::cast))
    }
}

ast_node!(DeclTarget, T![decl_target]);

impl DeclTarget {
    pub fn name(&self) -> Option<Ident> {
        self.0.first_child().and_then(Ident::cast)
    }

    pub fn modifier(&self) -> Option<DeclModifier> {
        match self.0.last_token() {
            Some(token) => DeclModifier::cast(token),
            None => None,
        }
    }
}

pub enum DeclModifier {
    Const,
    Close,
}

impl DeclModifier {
    fn cast(token: &SyntaxToken) -> Option<Self> {
        Some(match token.kind() {
            T![const] => Self::Const,
            T![close] => Self::Close,
            _ => return None,
        })
    }
}

ast_node!(Global, T![global_stmt]);

impl Global {
    pub fn function(&self) -> Option<Func> {
        self.0.first_child().and_then(Func::cast)
    }

    pub fn default_const(&self) -> bool {
        for tok in self.0.children_with_tokens() {
            if let NodeOrToken::Token(t) = tok {
                match t.kind() {
                    T![global] => continue,
                    T![const] => return true,
                    _ => return false,
                }
            } else {
                return false;
            }
        }
        false
    }

    pub fn is_star(&self) -> bool {
        self.0
            .children_with_tokens()
            .any(|or| matches!(or, NodeOrToken::Token(t) if t.kind() == T![*]))
    }

    pub fn targets(&self) -> Option<impl Iterator<Item = GlobalTarget> + '_> {
        Some(
            self.0
                .first_child()?
                .children()
                .filter_map(GlobalTarget::cast),
        )
    }

    /// Iterator over the optional initializer expressions in
    /// `global X, Y = e1, e2`. Returns `None` when there is no
    /// initializer (single child: the assign_list, or — for the
    /// function form — the Func node).
    pub fn values(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        let mut nodes = self.0.children();
        nodes.next()?;
        let expr_list = nodes.last()?;
        Some(expr_list.children().filter_map(Expr::cast))
    }
}

ast_node!(GlobalTarget, T![global_target]);

impl GlobalTarget {
    pub fn name(&self) -> Option<Ident> {
        self.0.first_child().and_then(Ident::cast)
    }

    pub fn is_const(&self) -> bool {
        self.0
            .last_token()
            .map(|t| t.kind() == T![const])
            .unwrap_or(false)
    }
}

ast_node!(Literal, T![literal_expr]);

impl Literal {
    pub fn value(&self, interner: &TokenInterner) -> Option<LiteralValue> {
        let token = self.0.first_token()?;
        let s = token.resolve_text(interner);

        Some(match token.kind() {
            T![nil] => LiteralValue::Nil,
            T![true] => LiteralValue::Bool(true),
            T![false] => LiteralValue::Bool(false),
            T![int] => LiteralValue::Int(lit::parse_int(s).ok()?),
            T![hex_int] => LiteralValue::Int(lit::parse_hex_int(s).ok()?),
            T![float] => LiteralValue::Float(lit::parse_float(s).ok()?),
            T![hex_float] => LiteralValue::Float(lit::parse_hex_float(s)?),
            T![string] => LiteralValue::String(lit::parse_string(s)),
            T![long_string] => LiteralValue::String(lit::parse_long_string(s)),
            T![ident] => LiteralValue::String(s.as_bytes().to_vec()),
            _ => unreachable!(),
        })
    }
}

pub enum LiteralValue {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Vec<u8>),
}

ast_node!(Assign, T![assign_stmt]);

impl Assign {
    pub fn targets(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        Some(self.0.first_child()?.children().filter_map(Expr::cast))
    }

    pub fn values(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        Some(self.0.last_child()?.children().filter_map(Expr::cast))
    }
}

ast_node!(Ident, T![ident]);

impl Ident {
    pub fn name<'a>(&self, interner: &'a TokenInterner) -> Option<&'a str> {
        self.0
            .first_token()
            .map(|token| token.resolve_text(interner))
    }
}

ast_node!(PrefixOp, T![prefix_op]);

impl PrefixOp {
    pub fn op(&self) -> Option<PrefixOperator> {
        self.0.first_token().and_then(PrefixOperator::cast)
    }

    pub fn rhs(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }
}

pub enum PrefixOperator {
    None,
    Neg,
    Not,
    Len,
    BitNot,
}

impl PrefixOperator {
    fn cast(token: &SyntaxToken) -> Option<Self> {
        Some(match token.kind() {
            T![+] => Self::None,
            T![-] => Self::Neg,
            T![~] => Self::BitNot,
            T![#] => Self::Len,
            T![not] => Self::Not,
            _ => panic!(),
        })
    }
}

ast_node!(BinaryOp, T![bin_op]);

impl BinaryOp {
    pub fn op(&self) -> Option<BinaryOperator> {
        self.0
            .children_with_tokens()
            .nth(1)
            .and_then(|or| match or {
                NodeOrToken::Node(_) => unreachable!(),
                NodeOrToken::Token(t) => BinaryOperator::cast(t),
            })
    }

    pub fn lhs(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn rhs(&self) -> Option<Expr> {
        self.0.last_child().and_then(Expr::cast)
    }
}

#[derive(PartialEq, Eq)]
pub enum BinaryOperator {
    And,
    Or,
    Add,
    Sub,
    Mul,
    Div,
    IntDiv,
    Exp,
    Mod,
    BitAnd,
    BitOr,
    LShift,
    RShift,
    Eq,
    BitXor,
    NEq,
    LEq,
    GEq,
    Gt,
    Lt,
    Property,
    Method,
    Concat,
}

impl BinaryOperator {
    fn cast(token: &SyntaxToken) -> Option<Self> {
        Some(match token.kind() {
            T![and] => Self::And,
            T![or] => Self::Or,
            T![+] => Self::Add,
            T![-] => Self::Sub,
            T![*] => Self::Mul,
            T![/] => Self::Div,
            T![D/] => Self::IntDiv,
            T![^] => Self::Exp,
            T![%] => Self::Mod,
            T![&] => Self::BitAnd,
            T![|] => Self::BitOr,
            T![<<] => Self::LShift,
            T![>>] => Self::RShift,
            T![==] => Self::Eq,
            T![~] => Self::BitXor,
            T![~=] => Self::NEq,
            T![<=] => Self::LEq,
            T![>=] => Self::GEq,
            T![>] => Self::Gt,
            T![<] => Self::Lt,
            T![.] => Self::Property,
            T![:] => Self::Method,
            T![..] => Self::Concat,
            _ => return None,
        })
    }
}

ast_node!(Index, T![index]);

impl Index {
    pub fn target(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn index(&self) -> Option<Expr> {
        self.0.last_child().and_then(Expr::cast)
    }
}

ast_node!(FuncCall, T![func_call]);

impl FuncCall {
    pub fn target(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn args(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        Some(self.0.last_child()?.children().filter_map(Expr::cast))
    }
}

ast_node!(Func, T![func_stmt]);

impl Func {
    pub fn name<'a>(
        &self,
        interner: &'a TokenInterner,
    ) -> Option<SyntaxText<'_, 'a, TokenInterner, SyntaxKind>> {
        Some(self.0.first_child()?.resolve_text(interner))
    }

    pub fn target(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn args(&self) -> Option<impl Iterator<Item = Ident> + '_> {
        Some(self.0.children().nth(1)?.children().filter_map(Ident::cast))
    }

    pub fn block(&self) -> Option<impl Iterator<Item = Stmt> + '_> {
        Some(self.0.last_child()?.children().filter_map(Stmt::cast))
    }
}

ast_node!(FuncExpr, T![func_expr]);

impl FuncExpr {
    pub fn args(&self) -> Option<impl Iterator<Item = Ident> + '_> {
        Some(self.0.first_child()?.children().filter_map(Ident::cast))
    }

    pub fn block(&self) -> Option<impl Iterator<Item = Stmt> + '_> {
        Some(self.0.last_child()?.children().filter_map(Stmt::cast))
    }
}

ast_node!(TableArray, T![table_array_elem]);

impl TableArray {
    pub fn value(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }
}

ast_node!(TableMap, T![table_map_elem]);

impl TableMap {
    pub fn field(&self) -> Option<Ident> {
        self.0.first_child().and_then(Ident::cast)
    }

    pub fn value(&self) -> Option<Expr> {
        self.0.last_child().and_then(Expr::cast)
    }
}

ast_node!(TableGeneric, T![table_generic_elem]);

impl TableGeneric {
    pub fn index(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn value(&self) -> Option<Expr> {
        self.0.last_child().and_then(Expr::cast)
    }
}

ast_node!(Table, T![table_expr]);

impl Table {
    pub fn entries(&self) -> impl Iterator<Item = TableEntry> + '_ {
        self.0.children().filter_map(TableEntry::cast)
    }
}

pub enum TableEntry {
    Array(TableArray),
    Map(TableMap),
    Generic(TableGeneric),
}

impl TableEntry {
    fn cast(node: &SyntaxNode) -> Option<Self> {
        Some(match node.kind() {
            T![table_array_elem] => TableArray::cast(node).map(Self::Array)?,
            T![table_map_elem] => TableMap::cast(node).map(Self::Map)?,
            T![table_generic_elem] => TableGeneric::cast(node).map(Self::Generic)?,
            _ => panic!(),
        })
    }
}

ast_node!(Break, T![break_stmt]);

ast_node!(Return, T![return_stmt]);

impl Return {
    pub fn exprs(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        Some(self.0.first_child()?.children().filter_map(Expr::cast))
    }
}

ast_node!(Do, T![do_stmt]);

impl Do {
    pub fn stmts(&self) -> Option<impl Iterator<Item = Stmt> + '_> {
        Some(self.0.first_child()?.children().filter_map(Stmt::cast))
    }
}

ast_node!(While, T![while_stmt]);

impl While {
    pub fn cond(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn block(&self) -> Option<Do> {
        self.0.children().nth(1).and_then(Do::cast)
    }
}

ast_node!(Repeat, T![repeat_stmt]);

impl Repeat {
    pub fn cond(&self) -> Option<Expr> {
        self.0.last_child().and_then(Expr::cast)
    }

    pub fn block(&self) -> Option<impl Iterator<Item = Stmt> + '_> {
        Some(self.0.first_child()?.children().filter_map(Stmt::cast))
    }
}

ast_node!(If, T![if_stmt]);

impl If {
    pub fn cond(&self) -> Option<Expr> {
        self.0.first_child().and_then(Expr::cast)
    }

    pub fn stmts(&self) -> Option<impl Iterator<Item = Stmt> + '_> {
        Some(self.0.children().nth(1)?.children().filter_map(Stmt::cast))
    }

    pub fn else_chain(&self) -> Option<ElseChain> {
        self.0.last_child().and_then(ElseChain::cast)
    }
}

ast_node!(ElseChain, T![else_chain]);

impl ElseChain {
    pub fn else_block(&self) -> Option<impl Iterator<Item = Stmt> + '_> {
        let token = self.0.first_token()?;

        if token.kind() == T![else] {
            Some(self.0.first_child()?.children().filter_map(Stmt::cast))
        } else {
            None
        }
    }

    pub fn elseif_block(&self) -> Option<If> {
        let token = self.0.first_token()?;

        if token.kind() == T![elseif] {
            If::cast(self.0.first_child()?)
        } else {
            None
        }
    }
}

ast_node!(ForNum, T![for_num_stmt]);

impl ForNum {
    pub fn counter(&self) -> Option<(Ident, Expr)> {
        let mut children = self.0.children();
        let name = children.next().and_then(Ident::cast)?;
        let value = children.next().and_then(Expr::cast)?;
        Some((name, value))
    }

    pub fn end(&self) -> Option<Expr> {
        self.0.children().nth(2).and_then(Expr::cast)
    }

    pub fn step(&self) -> Option<Expr> {
        if self.0.children().count() > 4 {
            return self.0.children().nth(3).and_then(Expr::cast);
        }

        None
    }

    pub fn block(&self) -> Option<Do> {
        self.0.last_child().and_then(Do::cast)
    }
}

ast_node!(ForGen, T![for_gen_stmt]);

impl ForGen {
    pub fn targets(&self) -> Option<impl Iterator<Item = Ident> + '_> {
        Some(self.0.first_child()?.children().filter_map(Ident::cast))
    }

    pub fn values(&self) -> Option<impl Iterator<Item = Expr> + '_> {
        Some(self.0.children().nth(1)?.children().filter_map(Expr::cast))
    }

    pub fn block(&self) -> Option<Do> {
        self.0.last_child().and_then(Do::cast)
    }
}
