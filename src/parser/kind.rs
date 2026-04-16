use std::fmt::{self, Display};

use logos::{Lexer, Logos};

#[allow(clippy::manual_non_exhaustive)]
#[derive(Logos, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
#[repr(u32)]
pub enum SyntaxKind {
    Invalid,
    Tombstone,

    Eof,
    Root,
    BreakStmt,
    ReturnStmt,
    DoStmt,
    WhileStmt,
    RepeatStmt,
    StmtList,
    IfStmt,
    ElseChain,
    ForNumStmt,
    ForGenStmt,
    FuncStmt,
    FuncArgs,
    Expr,
    VarArgExpr,
    BinOp,
    FuncCall,
    Index,
    ExprList,
    DeclStmt,
    DeclTarget,
    FuncExpr,
    PrefixOp,
    TableExpr,
    TableArrayElem,
    TableMapElem,
    TableGenericElem,
    AssignStmt,
    LiteralExpr,
    AssignList,
    Label,
    MethodCall,

    #[regex(r"[ \n\t\f\r]+", logos::skip)]
    Whitespace,

    #[regex("--", skip_comment)]
    Comment,

    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    #[token("%")]
    Percent,

    #[token("^")]
    Caret,

    #[token("#")]
    Hash,

    #[token("&")]
    Ampersand,

    #[token("|")]
    Pipe,

    #[token("~")]
    Tilde,

    #[token("<<")]
    DLAngle,

    #[token(">>")]
    DRAngle,

    #[token("==")]
    Eq,

    #[token("~=")]
    NotEq,

    #[token("<=")]
    LEq,

    #[token(">=")]
    GEq,

    #[token("<")]
    LAngle,

    #[token(">")]
    RAngle,

    #[token("=")]
    Assign,

    #[token("//")]
    DSlash,

    #[token(".")]
    Dot,

    #[token("..")]
    DDot,

    #[token("local")]
    Local,

    #[token("function")]
    Function,

    #[token("end")]
    End,

    #[token("in")]
    In,

    #[token("then")]
    Then,

    #[token("break")]
    Break,

    #[token("for")]
    For,

    #[token("do")]
    Do,

    #[token("until")]
    Until,

    #[token("else")]
    Else,

    #[token("while")]
    While,

    #[token("elseif")]
    ElseIf,

    #[token("if")]
    If,

    #[token("repeat")]
    Repeat,

    #[token("return")]
    Return,

    #[token("not")]
    Not,

    #[token("or")]
    Or,

    #[token("and")]
    And,

    #[token("goto")]
    Goto,

    #[token("<const>")]
    Const,

    #[token("<close>")]
    Close,

    #[token("nil")]
    Nil,

    #[token("true")]
    True,

    #[token("false")]
    False,

    #[regex(r#""(\\[\\"]|[^"])*""#)]
    #[regex(r#"'(\\[\\']|[^'])*'"#)]
    String,

    #[regex(r"\[=*\[", long_string)]
    LongString,

    #[regex(r"[0-9]+", priority = 3)]
    Int,

    #[regex(r"0x[0-9a-fA-F]+", priority = 7)]
    HexInt,

    #[regex(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?")]
    Float,

    #[regex(r"0x[0-9a-fA-F]+(\.[0-9a-fA-F]+)?([pP][+-][0-9a-fA-F]+)?")]
    HexFloat,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", priority = 3)]
    Ident,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token("{")]
    LCurly,

    #[token("}")]
    RCurly,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token(":")]
    Colon,

    #[token("::")]
    DColon,

    #[token(",")]
    Comma,

    #[token("...")]
    TDot,

    #[token(";")]
    Semicolon,
}

impl SyntaxKind {
    pub fn is_trivia(self) -> bool {
        matches!(self, SyntaxKind::Whitespace | SyntaxKind::Comment)
    }
}

fn long_string(lexer: &mut Lexer<SyntaxKind>) {
    let delim_len = lexer.slice().len();
    let rem = lexer.remainder();

    for (i, _) in rem.char_indices() {
        if is_long_delimiter(&rem[i..i + delim_len], ']') {
            lexer.bump(i + delim_len);
            return;
        }
    }

    unreachable!()
}

fn skip_comment(lexer: &mut Lexer<SyntaxKind>) -> logos::Skip {
    let rem = lexer.remainder();

    if let Some(delim_len) = starts_with_long_delimiter(rem, '[') {
        lexer.bump(delim_len);
        skip_long_comment(lexer, delim_len);
        logos::Skip
    } else {
        for (i, _) in rem.char_indices() {
            let curr = &rem[i..];
            if curr.starts_with("\r\n") {
                lexer.bump(i - 1);
                return logos::Skip;
            }

            if curr.starts_with('\n') {
                lexer.bump(i);
                return logos::Skip;
            }
        }

        unreachable!();
    }
}

fn skip_long_comment(lexer: &mut Lexer<SyntaxKind>, delim_len: usize) {
    let rem = lexer.remainder();

    for (i, _) in rem.char_indices() {
        if is_long_delimiter(&rem[i..i + delim_len], ']') {
            lexer.bump(i + delim_len);
            return;
        }
    }

    unreachable!()
}

fn starts_with_long_delimiter(slice: &str, delim: char) -> Option<usize> {
    if !slice.starts_with("[[") && !slice.starts_with("[=]") {
        return None;
    }

    for (i, _) in slice.char_indices() {
        if is_long_delimiter(&slice[..i], delim) {
            return Some(i);
        }
    }

    None
}

fn is_long_delimiter(slice: &str, delim: char) -> bool {
    if slice.len() < 2 || !slice.starts_with(delim) || !slice.ends_with(delim) {
        return false;
    }

    slice.chars().filter(|c| *c == '=').count() + 2 == slice.len()
}

#[macro_export]
macro_rules! T {
    [invalid] => { $crate::parser::kind::SyntaxKind::Invalid };
    [tombstone] => { $crate::parser::kind::SyntaxKind::Tombstone };
    [eof] => { $crate::parser::kind::SyntaxKind::Eof };
    [root] => { $crate::parser::kind::SyntaxKind::Root };
    [break_stmt] => { $crate::parser::kind::SyntaxKind::BreakStmt };
    [return_stmt] => { $crate::parser::kind::SyntaxKind::ReturnStmt };
    [do_stmt] => { $crate::parser::kind::SyntaxKind::DoStmt };
    [while_stmt] => { $crate::parser::kind::SyntaxKind::WhileStmt };
    [repeat_stmt] => { $crate::parser::kind::SyntaxKind::RepeatStmt };
    [stmt_list] => { $crate::parser::kind::SyntaxKind::StmtList };
    [if_stmt] => { $crate::parser::kind::SyntaxKind::IfStmt };
    [else_chain] => { $crate::parser::kind::SyntaxKind::ElseChain };
    [for_num_stmt] => { $crate::parser::kind::SyntaxKind::ForNumStmt };
    [for_gen_stmt] => { $crate::parser::kind::SyntaxKind::ForGenStmt };
    [func_stmt] => { $crate::parser::kind::SyntaxKind::FuncStmt };
    [func_args] => { $crate::parser::kind::SyntaxKind::FuncArgs };
    [expr] => { $crate::parser::kind::SyntaxKind::Expr };
    [vararg_expr] => { $crate::parser::kind::SyntaxKind::VarArgExpr };
    [bin_op] => { $crate::parser::kind::SyntaxKind::BinOp };
    [func_call] => { $crate::parser::kind::SyntaxKind::FuncCall };
    [index] => { $crate::parser::kind::SyntaxKind::Index };
    [expr_list] => { $crate::parser::kind::SyntaxKind::ExprList };
    [decl_stmt] => { $crate::parser::kind::SyntaxKind::DeclStmt };
    [decl_target] => { $crate::parser::kind::SyntaxKind::DeclTarget };
    [func_expr] => { $crate::parser::kind::SyntaxKind::FuncExpr };
    [prefix_op] => { $crate::parser::kind::SyntaxKind::PrefixOp };
    [table_expr] => { $crate::parser::kind::SyntaxKind::TableExpr };
    [table_array_elem] => { $crate::parser::kind::SyntaxKind::TableArrayElem };
    [table_map_elem] => { $crate::parser::kind::SyntaxKind::TableMapElem };
    [table_generic_elem] => { $crate::parser::kind::SyntaxKind::TableGenericElem };
    [assign_stmt] => { $crate::parser::kind::SyntaxKind::AssignStmt };
    [literal_expr] => { $crate::parser::kind::SyntaxKind::LiteralExpr };
    [ident] => { $crate::parser::kind::SyntaxKind::Ident };
    [assign_list] => { $crate::parser::kind::SyntaxKind::AssignList };
    [label] => { $crate::parser::kind::SyntaxKind::Label };
    [method_call] => { $crate::parser::kind::SyntaxKind::MethodCall };
    [+] => { $crate::parser::kind::SyntaxKind::Plus };
    [-] => { $crate::parser::kind::SyntaxKind::Minus };
    [*] => { $crate::parser::kind::SyntaxKind::Star };
    [/] => { $crate::parser::kind::SyntaxKind::Slash };
    [%] => { $crate::parser::kind::SyntaxKind::Percent };
    [^] => { $crate::parser::kind::SyntaxKind::Caret };
    [#] => { $crate::parser::kind::SyntaxKind::Hash };
    [&] => { $crate::parser::kind::SyntaxKind::Ampersand };
    [|] => { $crate::parser::kind::SyntaxKind::Pipe };
    [~] => { $crate::parser::kind::SyntaxKind::Tilde };
    [<<] => { $crate::parser::kind::SyntaxKind::DLAngle };
    [>>] => { $crate::parser::kind::SyntaxKind::DRAngle };
    [==] => { $crate::parser::kind::SyntaxKind::Eq };
    [~=] => { $crate::parser::kind::SyntaxKind::NotEq };
    [<=] => { $crate::parser::kind::SyntaxKind::LEq };
    [>=] => { $crate::parser::kind::SyntaxKind::GEq };
    [<] => { $crate::parser::kind::SyntaxKind::LAngle };
    [>] => { $crate::parser::kind::SyntaxKind::RAngle };
    [=] => { $crate::parser::kind::SyntaxKind::Assign };
    [D/] => { $crate::parser::kind::SyntaxKind::DSlash };
    [local] => { $crate::parser::kind::SyntaxKind::Local };
    [function] => { $crate::parser::kind::SyntaxKind::Function };
    [end] => { $crate::parser::kind::SyntaxKind::End };
    [in] => { $crate::parser::kind::SyntaxKind::In };
    [then] => { $crate::parser::kind::SyntaxKind::Then };
    [break] => { $crate::parser::kind::SyntaxKind::Break };
    [for] => { $crate::parser::kind::SyntaxKind::For };
    [do] => { $crate::parser::kind::SyntaxKind::Do };
    [until] => { $crate::parser::kind::SyntaxKind::Until };
    [else] => { $crate::parser::kind::SyntaxKind::Else };
    [while] => { $crate::parser::kind::SyntaxKind::While };
    [elseif] => { $crate::parser::kind::SyntaxKind::ElseIf };
    [if] => { $crate::parser::kind::SyntaxKind::If };
    [repeat] => { $crate::parser::kind::SyntaxKind::Repeat };
    [return] => { $crate::parser::kind::SyntaxKind::Return };
    [not] => { $crate::parser::kind::SyntaxKind::Not };
    [or] => { $crate::parser::kind::SyntaxKind::Or };
    [and] => { $crate::parser::kind::SyntaxKind::And };
    [goto] => { $crate::parser::kind::SyntaxKind::Goto };
    [const] => { $crate::parser::kind::SyntaxKind::Const };
    [close] => { $crate::parser::kind::SyntaxKind::Close };
    [nil] => { $crate::parser::kind::SyntaxKind::Nil };
    [true] => { $crate::parser::kind::SyntaxKind::True };
    [false] => { $crate::parser::kind::SyntaxKind::False };
    [string] => { $crate::parser::kind::SyntaxKind::String };
    [long_string] => { $crate::parser::kind::SyntaxKind::LongString };
    [int] => { $crate::parser::kind::SyntaxKind::Int };
    [hex_int] => { $crate::parser::kind::SyntaxKind::HexInt };
    [float] => { $crate::parser::kind::SyntaxKind::Float };
    [hex_float] => { $crate::parser::kind::SyntaxKind::HexFloat };
    ['('] => { $crate::parser::kind::SyntaxKind::LParen };
    [')'] => { $crate::parser::kind::SyntaxKind::RParen };
    ['{'] => { $crate::parser::kind::SyntaxKind::LCurly };
    ['}'] => { $crate::parser::kind::SyntaxKind::RCurly };
    ['['] => { $crate::parser::kind::SyntaxKind::LBracket };
    [']'] => { $crate::parser::kind::SyntaxKind::RBracket };
    [:] => { $crate::parser::kind::SyntaxKind::Colon };
    [::] => { $crate::parser::kind::SyntaxKind::DColon };
    [,] => { $crate::parser::kind::SyntaxKind::Comma };
    [.] => { $crate::parser::kind::SyntaxKind::Dot };
    [..] => { $crate::parser::kind::SyntaxKind::DDot };
    [...] => { $crate::parser::kind::SyntaxKind::TDot };
    [;] => { $crate::parser::kind::SyntaxKind::Semicolon };
    [whitespace] => { $crate::parser::kind::SyntaxKind::Whitespace };
    [comment] => { $crate::parser::kind::SyntaxKind::Comment };
}

impl Display for SyntaxKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                T![invalid] => "invalid",
                T![tombstone] => "tombstone",
                T![eof] => "eof",
                T![root] => "root",
                T![break_stmt] => "break_stmt",
                T![return_stmt] => "return_stmt",
                T![do_stmt] => "do_stmt",
                T![while_stmt] => "while_stmt",
                T![repeat_stmt] => "repeat_stmt",
                T![stmt_list] => "stmt_list",
                T![if_stmt] => "if_stmt",
                T![else_chain] => "else_chain",
                T![for_num_stmt] => "for_num_stmt",
                T![for_gen_stmt] => "for_gen_stmt",
                T![func_stmt] => "func_stmt",
                T![func_args] => "func_args",
                T![expr] => "expr",
                T![vararg_expr] => "vararg_expr",
                T![bin_op] => "bin_op",
                T![func_call] => "func_call",
                T![index] => "index",
                T![expr_list] => "expr_list",
                T![decl_stmt] => "decl_stmt",
                T![decl_target] => "decl_target",
                T![func_expr] => "func_expr",
                T![prefix_op] => "prefix_op",
                T![table_expr] => "table_expr",
                T![table_array_elem] => "table_array_elem",
                T![table_map_elem] => "table_map_elem",
                T![table_generic_elem] => "table_generic_elem",
                T![assign_stmt] => "assign_stmt",
                T![literal_expr] => "literal_expr",
                T![ident] => "ident",
                T![assign_list] => "assign_list",
                T![label] => "label",
                T![method_call] => "method_call",
                T![+] => "+",
                T![-] => "-",
                T![*] => "*",
                T![/] => "/",
                T![%] => "%",
                T![^] => "^",
                T![#] => "#",
                T![&] => "&",
                T![|] => "|",
                T![~] => "~",
                T![<<] => "<<",
                T![>>] => ">>",
                T![==] => "==",
                T![~=] => "~=",
                T![<=] => "<=",
                T![>=] => ">=",
                T![<] => "<",
                T![>] => ">",
                T![=] => "=",
                T![D/] => "D/",
                T![local] => "local",
                T![function] => "function",
                T![end] => "end",
                T![in] => "in",
                T![then] => "then",
                T![break] => "break",
                T![for] => "for",
                T![do] => "do",
                T![until] => "until",
                T![else] => "else",
                T![while] => "while",
                T![elseif] => "elseif",
                T![if] => "if",
                T![repeat] => "repeat",
                T![return] => "return",
                T![not] => "not",
                T![or] => "or",
                T![and] => "and",
                T![goto] => "goto",
                T![const] => "const",
                T![close] => "close",
                T![nil] => "nil",
                T![true] => "true",
                T![false] => "false",
                T![string] => "string",
                T![long_string] => "long_string",
                T![int] => "int",
                T![hex_int] => "hex_int",
                T![float] => "float",
                T![hex_float] => "hex_float",
                T!['('] => "'('",
                T![')'] => "')'",
                T!['{'] => "'{'",
                T!['}'] => "'}'",
                T!['['] => "'['",
                T![']'] => "']'",
                T![:] => ":",
                T![::] => "::",
                T![,] => ",",
                T![.] => ".",
                T![..] => "..",
                T![...] => "...",
                T![;] => ";",
                T![whitespace] => "whitespace",
                T![comment] => "comment",
            }
        )
    }
}
