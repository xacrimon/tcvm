use super::{
    Parser,
    kind::SyntaxKind,
    machinery::{CompletedMarker, Marker, token_is_expr_start},
};
use crate::{
    T,
    parser::machinery::{
        CALL_BINDING_POWER, INDEX_BINDING_POWER, infix_binding_power, prefix_binding_power,
        token_is_literal, token_is_unary_op,
    },
};

const STATEMENT_RECOVERY: &[SyntaxKind] = &[
    T![do],
    T![while],
    T![repeat],
    T![if],
    T![for],
    T![return],
    T![break],
    T![function],
    T![local],
    T![global],
];

impl<'cache, 'source> Parser<'cache, 'source> {
    pub(super) fn r_items(&mut self) {
        while self.at() != T![eof] {
            if self.r_stmt().is_none() {
                break;
            }
        }
    }

    fn r_stmt(&mut self) -> Option<CompletedMarker> {
        let marker = match self.at() {
            T![do] => self.r_do(),
            T![while] => self.r_while(),
            T![repeat] => self.r_repeat(),
            T![if] => self.r_if(T![if]),
            T![for] => self.r_for(),
            T![return] => self.r_return(),
            T![break] => self.r_break(),
            T![function] => self.r_func(false),
            T![local] => self.r_decl(),
            T![global] => self.r_global(),
            T![::] => self.r_label(),
            T![goto] => self.r_goto(),
            T![ident] | T!['('] => self.r_maybe_assign(),
            T![;] => self.r_semicolon(),
            _ => None,
        };

        if marker.is_none() {
            if self.at() == T![eof] {
                return None;
            }

            let span = self.error_eat_until(STATEMENT_RECOVERY);
            let source = self.source(span);
            let error = self
                .new_error()
                .with_message("expected a statement")
                .with_label(
                    self.new_label()
                        .with_message(format!("expected a statement but got \"{}\"", source,)),
                )
                .finish();

            self.report(error);
            return self.r_stmt();
        }

        marker
    }

    fn r_label(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![label]);
        self.expect(T![::]);
        self.r_ident();
        self.expect(T![::]);
        Some(marker.complete(self))
    }

    fn r_goto(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![goto]);
        self.expect(T![goto]);
        self.r_ident();
        Some(marker.complete(self))
    }

    fn r_expr_list(&mut self) {
        let marker = self.start(T![expr_list]);

        while token_is_expr_start(self.at()) {
            self.r_expr();
            if self.at() != T![,] {
                break;
            }

            self.expect(T![,]);
        }

        marker.complete(self);
    }

    fn r_expr(&mut self) -> Option<CompletedMarker> {
        self.r_expr_inner(0)
    }

    fn r_expr_inner(&mut self, min_bp: i32) -> Option<CompletedMarker> {
        let mut lhs = self.r_expr_lhs()?;

        loop {
            let t = self.at();

            if t == T![:] {
                let n = lhs.precede(self, T![method_call]);
                lhs = self.r_method_call(n)?;
                continue;
            }

            if t == T!['('] && CALL_BINDING_POWER >= min_bp {
                let n = lhs.precede(self, T![func_call]);
                let _rhs = self.r_func_call_args()?;
                lhs = n.complete(self);
                continue;
            }

            if t == T!['['] && INDEX_BINDING_POWER >= min_bp {
                let n = lhs.precede(self, T![index]);
                self.expect(T!['[']);
                let _rhs = self.r_expr()?;
                self.expect(T![']']);
                lhs = n.complete(self);
                continue;
            }

            if let Some((l_bp, r_bp)) = infix_binding_power(t) {
                if l_bp < min_bp {
                    break;
                }

                let n = lhs.precede(self, T![bin_op]);
                self.expect(t);

                if T![.] == t {
                    let _rhs = self.r_literal()?;
                } else {
                    let _rhs = self.r_expr_inner(r_bp)?;
                }

                lhs = n.complete(self);
                continue;
            }

            break;
        }

        Some(lhs)
    }

    fn r_expr_lhs(&mut self) -> Option<CompletedMarker> {
        match self.at() {
            T![ident] => self.r_ident(),
            T![...] => self.r_vararg(),
            T!['{'] => self.r_table(),
            T!['('] => self.r_paren(),
            T![function] => self.r_func(true),
            t if token_is_unary_op(t) => self.r_expr_unary(),
            t if token_is_literal(t) => self.r_literal(),
            _ => None,
        }
    }

    fn r_expr_unary(&mut self) -> Option<CompletedMarker> {
        let n = self.start(T![prefix_op]);
        let op = self.at();
        self.expect(op);
        let ((), r_bp) = prefix_binding_power(op);
        let _rhs = self.r_expr_inner(r_bp);
        Some(n.complete(self))
    }

    fn r_method_call(&mut self, marker: Marker) -> Option<CompletedMarker> {
        self.expect(T![:]);
        self.r_ident()?;
        self.r_func_call_args()?;
        Some(marker.complete(self))
    }

    fn r_ident(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![ident]);
        self.expect(T![ident]);
        Some(marker.complete(self))
    }

    fn r_vararg(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![vararg_expr]);
        self.expect(T![...]);
        Some(marker.complete(self))
    }

    fn r_paren(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![expr]);
        self.expect(T!['(']);
        let _rhs = self.r_expr()?;
        self.expect(T![')']);
        Some(marker.complete(self))
    }

    fn r_literal(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![literal_expr]);
        let kind = self.at();
        self.expect(kind);
        Some(marker.complete(self))
    }

    fn r_do(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![do_stmt]);
        self.expect(T![do]);
        self.r_block(&|t| t == T![end]);
        self.expect(T![end]);
        Some(marker.complete(self))
    }

    fn r_while(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![while_stmt]);
        self.expect(T![while]);
        self.r_expr();
        self.r_do();
        Some(marker.complete(self))
    }

    fn r_repeat(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![repeat_stmt]);
        self.expect(T![repeat]);
        self.r_block(&|t| t == T![until]);
        self.expect(T![until]);
        self.r_expr();
        Some(marker.complete(self))
    }

    fn r_if(&mut self, if_kind: SyntaxKind) -> Option<CompletedMarker> {
        let marker = self.start(T![if_stmt]);
        self.expect(if_kind);
        self.r_expr();
        self.expect(T![then]);
        self.r_block(&|t| matches!(t, T![end] | T![elseif] | T![else]));

        match self.at() {
            T![end] => {
                self.expect(T![end]);
            }
            T![elseif] | T![else] => {
                self.r_else();
            }
            t => {
                let error = self
                    .new_error()
                    .with_message("unexpected token")
                    .with_label(self.new_label().with_message(format!(
                        "expected token one of [{}, {}, {}] but found {}",
                        T![end],
                        T![elseif],
                        T![else],
                        t,
                    )))
                    .finish();

                self.report(error);
                return None;
            }
        }

        Some(marker.complete(self))
    }

    fn r_else(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![else_chain]);

        match self.at() {
            T![else] => {
                self.expect(T![else]);
                self.r_block(&|t| t == T![end]);
                self.expect(T![end]);
            }
            T![elseif] => {
                self.r_if(T![elseif]);
            }
            _ => unreachable!(),
        }

        Some(marker.complete(self))
    }

    fn r_for(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![tombstone]);
        self.expect(T![for]);
        let assign_list_marker = self.start(T![assign_list]);
        self.r_ident();

        if self.at() == T![=] {
            assign_list_marker.abandon(self);
            self.r_num_for(marker)
        } else {
            self.r_gen_for(marker, assign_list_marker)
        }
    }

    fn r_num_for(&mut self, marker: Marker) -> Option<CompletedMarker> {
        self.expect(T![=]);
        self.r_expr();
        self.expect(T![,]);
        self.r_expr();
        if self.at() == T![,] {
            self.expect(T![,]);
            self.r_expr();
        }

        self.r_do();
        Some(marker.retype(self, T![for_num_stmt]).complete(self))
    }

    fn r_gen_for(&mut self, marker: Marker, list_marker: Marker) -> Option<CompletedMarker> {
        while self.at() == T![,] {
            self.expect(T![,]);
            self.r_ident();
        }

        list_marker.complete(self);
        self.expect(T![in]);
        self.r_expr_list();
        self.r_do();
        Some(marker.retype(self, T![for_gen_stmt]).complete(self))
    }

    fn r_return(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![return_stmt]);
        self.expect(T![return]);
        self.r_expr_list();
        Some(marker.complete(self))
    }

    fn r_break(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![break_stmt]);
        self.expect(T![break]);
        Some(marker.complete(self))
    }

    fn r_block(&mut self, stop: &dyn Fn(SyntaxKind) -> bool) -> Option<CompletedMarker> {
        let marker = self.start(T![stmt_list]);
        while !stop(self.at()) {
            self.r_stmt();
        }

        Some(marker.complete(self))
    }

    fn r_func_call_args(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![func_args]);
        self.expect(T!['(']);

        loop {
            match self.at() {
                T![')'] => {
                    self.expect(T![')']);
                    break;
                }
                _ => {
                    self.r_expr();
                }
            }

            if self.at() == T![,] {
                self.expect(T![,]);
            } else {
                self.expect(T![')']);
                break;
            }
        }

        Some(marker.complete(self))
    }

    fn r_func(&mut self, expr: bool) -> Option<CompletedMarker> {
        let kind = if expr { T![func_expr] } else { T![func_stmt] };
        let marker = self.start(kind);
        self.expect(T![function]);

        if !expr {
            self.r_simple_expr(false);
        }

        self.r_func_def_args();
        self.r_block(&|t| t == T![end]);
        self.expect(T![end]);
        Some(marker.complete(self))
    }

    fn r_func_def_args(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![func_args]);
        self.expect(T!['(']);

        loop {
            match self.at() {
                T![')'] => {
                    self.expect(T![')']);
                    break;
                }
                T![...] => {
                    self.expect(T![...]);
                }
                T![ident] => {
                    self.r_ident();
                }
                t => {
                    let error = self
                        .new_error()
                        .with_message("unexpected token")
                        .with_label(self.new_label().with_message(format!(
                            "expected token one of [{}, {}, {}] but found {}",
                            T![')'],
                            T![...],
                            T![ident],
                            t,
                        )))
                        .finish();

                    self.report(error);

                    return None;
                }
            }

            if self.at() == T![,] {
                self.expect(T![,]);
            } else {
                self.expect(T![')']);
                break;
            }
        }

        Some(marker.complete(self))
    }

    fn r_semicolon(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![;]);
        self.expect(T![;]);
        Some(marker.complete(self))
    }

    fn r_table(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![table_expr]);
        self.expect(T!['{']);

        loop {
            match self.at() {
                T!['}'] => {
                    self.expect(T!['}']);
                    break;
                }
                _ => {
                    self.r_table_elem();
                }
            }

            let t = self.at();
            if t == T![,] || t == T![;] {
                self.expect(t);
            } else {
                self.expect(T!['}']);
                break;
            }
        }

        Some(marker.complete(self))
    }

    fn r_table_elem(&mut self) -> Option<CompletedMarker> {
        match self.at() {
            T![ident] if self.peek() == Some(T![=]) => self.r_table_elem_map(),
            T!['['] => self.r_table_elem_generic(),
            t if token_is_expr_start(t) => self.r_table_elem_array(),
            t => {
                let error = self
                    .new_error()
                    .with_message("unexpected token")
                    .with_label(self.new_label().with_message(format!(
                        "expected token one of [{}, {}, expr] but found {}",
                        T![ident],
                        T!['['],
                        t,
                    )))
                    .finish();

                self.report(error);
                None
            }
        }
    }

    fn r_table_elem_array(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![table_array_elem]);
        self.r_expr();
        Some(marker.complete(self))
    }

    fn r_table_elem_map(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![table_map_elem]);
        self.r_ident();
        self.expect(T![=]);
        self.r_expr();
        Some(marker.complete(self))
    }

    fn r_table_elem_generic(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![table_generic_elem]);
        self.expect(T!['[']);
        self.r_expr();
        self.expect(T![']']);
        self.expect(T![=]);
        self.r_expr();
        Some(marker.complete(self))
    }

    fn r_simple_expr(&mut self, allow_call: bool) -> Option<CompletedMarker> {
        if self.at() == T!['('] {
            return self.r_expr();
        }

        let mut lhs = self.r_ident()?;

        loop {
            let t = self.at();

            if t == T!['('] && allow_call {
                let n = lhs.precede(self, T![func_call]);
                let _rhs = self.r_func_call_args()?;
                lhs = n.complete(self);
                continue;
            }

            if t == T!['['] {
                let n = lhs.precede(self, T![index]);
                self.expect(T!['[']);
                let _rhs = self.r_expr()?;
                self.expect(T![']']);
                lhs = n.complete(self);
                continue;
            }

            if t == T![.] || t == T![:] {
                let n = lhs.precede(self, T![bin_op]);
                self.expect(t);
                let _rhs = self.r_literal();
                lhs = n.complete(self);
                continue;
            }

            break;
        }

        Some(lhs)
    }

    fn r_maybe_assign(&mut self) -> Option<CompletedMarker> {
        let assign_marker = self.start(T![assign_stmt]);
        let assign_list_marker = self.start(T![assign_list]);
        let expr_marker = self.r_simple_expr(true);
        if matches!(self.at(), T![=] | T![,]) {
            self.r_assign(assign_marker, assign_list_marker)
        } else {
            assign_list_marker.abandon(self);
            assign_marker.abandon(self);
            expr_marker
        }
    }

    fn r_assign(&mut self, assign_marker: Marker, list_marker: Marker) -> Option<CompletedMarker> {
        while self.at() == T![,] {
            self.expect(T![,]);
            self.r_simple_expr(true);
        }

        list_marker.complete(self);
        self.expect(T![=]);
        self.r_expr_list();
        Some(assign_marker.complete(self))
    }

    fn r_decl(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![decl_stmt]);
        self.expect(T![local]);

        if self.at() == T![function] {
            self.r_func(false);
        } else {
            let assign_list_marker = self.start(T![assign_list]);
            self.r_decl_target();

            while self.at() == T![,] {
                self.expect(T![,]);
                self.r_decl_target();
            }

            assign_list_marker.complete(self);
            if self.at() == T![=] {
                self.expect(T![=]);
                self.r_expr_list();
            }
        }

        Some(marker.complete(self))
    }

    fn r_decl_target(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![decl_target]);
        self.r_ident();
        self.r_attrib();
        Some(marker.complete(self))
    }

    fn r_attrib(&mut self) {
        let t = self.at();

        if matches!(t, T![const] | T![close]) {
            self.expect(t);
        }
    }

    fn r_global(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![global_stmt]);
        self.expect(T![global]);

        if self.at() == T![function] {
            self.r_func(false);
        } else {
            self.r_global_attrib();

            if self.at() == T![*] {
                self.expect(T![*]);
            } else {
                let assign_list_marker = self.start(T![assign_list]);
                self.r_global_target();

                while self.at() == T![,] {
                    self.expect(T![,]);
                    self.r_global_target();
                }

                assign_list_marker.complete(self);

                if self.at() == T![=] {
                    self.expect(T![=]);
                    self.r_expr_list();
                }
            }
        }

        Some(marker.complete(self))
    }

    fn r_global_target(&mut self) -> Option<CompletedMarker> {
        let marker = self.start(T![global_target]);
        self.r_ident();
        self.r_global_attrib();
        Some(marker.complete(self))
    }

    fn r_global_attrib(&mut self) {
        let t = self.at();

        if t == T![const] {
            self.expect(t);
        } else if t == T![close] {
            let error = self
                .new_error()
                .with_message("global variables cannot be to-be-closed")
                .with_label(
                    self.new_label()
                        .with_message("'<close>' is not allowed on global declarations"),
                )
                .finish();
            self.report(error);
            self.expect(t);
        }
    }
}
