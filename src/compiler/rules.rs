use std::collections::HashMap;

use cstree::interning::TokenInterner;

use super::defs::{Chunk, ExprDesc, ExprKind, JumpList, RegisterIndex};
use super::{CompileError, CompileErrorKind, LineNumber};
use crate::dmm::{Gc, Mutation};
use crate::env::{LuaString, Prototype, Value};
use crate::instruction::{Instruction, UpValueDescriptor};
use crate::parser::syntax::{
    Assign, BinaryOp, BinaryOperator, Break, Decl, DeclModifier, Do, Expr, ForGen, ForNum, Func,
    FuncCall, FuncExpr, Goto, Ident, If, Index, Label, Literal, LiteralValue, MethodCall, PrefixOp,
    PrefixOperator, Repeat, Return, Root, Stmt, Table, TableEntry, While,
};
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Error helper
// ---------------------------------------------------------------------------

fn ice(msg: &'static str) -> CompileError {
    CompileError::internal(msg)
}

fn err(kind: CompileErrorKind, line: LineNumber) -> CompileError {
    CompileError {
        kind,
        line_number: line,
    }
}

// ---------------------------------------------------------------------------
// Variable tracking
// ---------------------------------------------------------------------------

struct VariableData {
    register: RegisterIndex,
    is_const: bool,
}

// ---------------------------------------------------------------------------
// Compilation context
// ---------------------------------------------------------------------------

struct Ctx<'gc, 'a> {
    interner: &'a TokenInterner,
    mc: &'a Mutation<'gc>,
    chunk: Chunk<'gc>,

    /// Stack of break-target labels for nested loops.
    control_end_label: Vec<u16>,

    /// Lexical scope stack: each frame maps variable names to register data.
    scope: Vec<HashMap<String, VariableData>>,
    /// Saved register counts per scope (for restoring on pop).
    scope_register_base: Vec<usize>,
    /// Registers that need CLOSE when scope is popped (to-be-closed variables).
    scope_close: Vec<Vec<RegisterIndex>>,

    /// Named labels for goto/label statements (name → label index).
    goto_labels: HashMap<String, u16>,

    /// Callback to resolve a name as an upvalue from the enclosing function.
    /// Returns the upvalue index if found.
    capture: Box<dyn FnMut(&str) -> Option<u8> + 'a>,
    /// Upvalue descriptors accumulated for this function.
    upvalue_desc: Vec<UpValueDescriptor>,
}

impl<'gc, 'a> Ctx<'gc, 'a> {
    fn emit(&mut self, instruction: Instruction) {
        self.chunk.tape.push(instruction);
    }

    fn alloc_register(&mut self) -> Result<RegisterIndex, CompileError> {
        if self.chunk.register_count >= 255 {
            return Err(err(CompileErrorKind::Registers, LineNumber(0)));
        }
        let reg = self.chunk.register_count as u8;
        self.chunk.register_count += 1;
        if self.chunk.register_count > self.chunk.max_register_count {
            self.chunk.max_register_count = self.chunk.register_count;
        }
        Ok(RegisterIndex(reg))
    }

    /// Use a destination hint register if provided, otherwise allocate a fresh one.
    fn dst_or_alloc(&mut self, dst: Option<RegisterIndex>) -> Result<RegisterIndex, CompileError> {
        match dst {
            Some(reg) => {
                // Ensure register counter is past this register
                if self.chunk.register_count <= reg.0 as usize {
                    self.chunk.register_count = reg.0 as usize + 1;
                    if self.chunk.register_count > self.chunk.max_register_count {
                        self.chunk.max_register_count = self.chunk.register_count;
                    }
                }
                Ok(reg)
            }
            None => self.alloc_register(),
        }
    }

    fn next_offset(&self) -> usize {
        self.chunk.tape.len()
    }

    fn new_label(&mut self) -> u16 {
        let idx = self.chunk.labels.len();
        self.chunk.labels.push(0);
        idx as u16
    }

    fn set_label(&mut self, label: u16, offset: usize) {
        self.chunk.labels[label as usize] = offset;
    }

    fn emit_jump(&mut self, label: u16) {
        let idx = self.next_offset();
        self.chunk.jump_patches.push((idx, label));
        self.emit(Instruction::JMP { offset: 0 });
    }

    fn emit_jump_instr(&mut self, label: u16, instr: Instruction) {
        let idx = self.next_offset();
        self.chunk.jump_patches.push((idx, label));
        self.emit(instr);
    }

    // ---------------------------------------------------------------------
    // Jump-list primitives (Lua 5.4-style true/false list patching)
    // ---------------------------------------------------------------------

    /// Emit a `JMP` with a placeholder offset and return its tape index so
    /// the caller can thread it through a `JumpList`.
    fn emit_unfilled_jmp(&mut self) -> usize {
        let idx = self.next_offset();
        self.emit(Instruction::JMP { offset: 0 });
        idx
    }

    /// Patch every jump in `list` so it targets the instruction at `target`.
    fn patch_to(&mut self, list: JumpList, target: usize) {
        for idx in list.jumps {
            let offset = target as i32 - (idx as i32 + 1);
            match &mut self.chunk.tape[idx] {
                Instruction::JMP { offset: o } => *o = offset,
                _ => panic!("jump-list entry is not a JMP"),
            }
        }
    }

    /// Patch every jump in `list` to the current tape position (the next
    /// instruction to be emitted).
    fn patch_to_here(&mut self, list: JumpList) {
        let target = self.next_offset();
        self.patch_to(list, target);
    }

    /// Emit a test + unfilled jump that fires when the value in `src` matches
    /// `jump_if_truthy`. Returns the tape index of the emitted JMP so the
    /// caller can append it to the appropriate list.
    fn emit_test_jump(&mut self, src: RegisterIndex, jump_if_truthy: bool) -> usize {
        // op_test skips iff `truthy != inverted`, so to make the JMP fire
        // when the value has the desired truthiness we set `inverted` equal
        // to that desired truthiness. Then skip iff truthy != desired → JMP
        // runs iff truthy == desired.
        self.emit(Instruction::TEST {
            src: src.0,
            inverted: jump_if_truthy,
        });
        self.emit_unfilled_jmp()
    }

    /// Swap the true/false lists of `expr` and flip the `inverted` flag on
    /// each list's associated control instruction (the CMP/TEST immediately
    /// before each pending JMP). Lets callers reuse a pending conditional
    /// jump when they need the opposite polarity — no extra instructions.
    fn negate_cond(&mut self, expr: &mut ExprDesc) {
        for &jmp_idx in expr
            .true_list
            .jumps
            .iter()
            .chain(expr.false_list.jumps.iter())
        {
            // Only conditional jumps have a preceding control instruction.
            // Unconditional JMPs (produced during list merges) have no test
            // instruction in front of them and are left untouched — callers
            // avoid negating such expressions.
            if jmp_idx == 0 {
                continue;
            }
            match &mut self.chunk.tape[jmp_idx - 1] {
                Instruction::LT { inverted, .. }
                | Instruction::LE { inverted, .. }
                | Instruction::EQ { inverted, .. }
                | Instruction::TEST { inverted, .. }
                | Instruction::TESTSET { inverted, .. } => {
                    *inverted = !*inverted;
                }
                _ => {}
            }
        }
        std::mem::swap(&mut expr.true_list, &mut expr.false_list);
    }

    /// Ensure `expr` has a pending jump in its true-list — i.e. one that
    /// fires exactly when the expression evaluates to truthy. Leaves
    /// fall-through as the falsy path.
    fn goiftrue(&mut self, expr: &mut ExprDesc) -> Result<(), CompileError> {
        match expr.kind {
            ExprKind::Jump => {
                // If the pending jumps already fire on truthy (true_list
                // populated), we're done — fall-through is the falsy path.
                // If they only fire on falsy, invert polarity so they move
                // into true_list.
                if expr.true_list.is_empty() && !expr.false_list.is_empty() {
                    self.negate_cond(expr);
                }
            }
            ExprKind::Reg(reg) => {
                let jmp = self.emit_test_jump(reg, true);
                expr.true_list.jumps.push(jmp);
            }
        }
        Ok(())
    }

    /// Ensure `expr` has a pending jump in its false-list — fires when the
    /// expression evaluates to falsy, fall-through is the truthy path.
    fn goiffalse(&mut self, expr: &mut ExprDesc) -> Result<(), CompileError> {
        match expr.kind {
            ExprKind::Jump => {
                if expr.false_list.is_empty() && !expr.true_list.is_empty() {
                    self.negate_cond(expr);
                }
            }
            ExprKind::Reg(reg) => {
                let jmp = self.emit_test_jump(reg, false);
                expr.false_list.jumps.push(jmp);
            }
        }
        Ok(())
    }

    /// Discharge an expression to a concrete register. If either jump list
    /// is non-empty, emit a fixup tail so every exit path materialises a
    /// boolean in `dst`:
    ///
    /// ```text
    ///   <value in dst — for Reg kinds; Jump kinds have no "fall-through"
    ///    value because the preceding CMP/TEST fully routes into the lists>
    ///   JMP end                          ; skip past fixup (Reg path only)
    ///   LFALSESKIP dst                   ; false_list patches here
    ///   LOAD        dst K(true)          ; true_list patches here
    /// end:
    /// ```
    ///
    /// For a Jump expression the fall-through at the moment of discharge is
    /// already the "truthy" outcome of the pending comparison, so we route
    /// that into `true_list` first by emitting an unconditional JMP and
    /// appending it. That reduces the pattern to the fixup tail alone.
    fn discharge_to_reg_mut(
        &mut self,
        expr: &mut ExprDesc,
        hint: Option<RegisterIndex>,
    ) -> Result<RegisterIndex, CompileError> {
        // For a plain Reg with no pending jumps, return the register where
        // the value already lives. The `hint` is advisory — honouring it
        // here would emit a MOVE that many callers then re-emit themselves,
        // flipping instruction order and inflating the frame size. Callers
        // that need the value in a specific register check the returned
        // register and MOVE themselves.
        if matches!(expr.kind, ExprKind::Reg(_)) && !expr.has_jumps() {
            if let ExprKind::Reg(reg) = expr.kind {
                return Ok(reg);
            }
        }

        let dst = match expr.kind {
            ExprKind::Reg(reg) => reg,
            ExprKind::Jump => self.dst_or_alloc(hint)?,
        };

        let is_jump = matches!(expr.kind, ExprKind::Jump);
        if is_jump {
            // Route the comparison's truthy fall-through into true_list so
            // the fixup tail below is the only materialisation site.
            let jmp = self.emit_unfilled_jmp();
            expr.true_list.jumps.push(jmp);
        }

        if expr.has_jumps() {
            // For a Reg-kind with pending jumps (e.g. value-context `and`/`or`
            // whose LHS short-circuited into the lists), the fall-through at
            // this point is the LHS's already-materialised value in `dst`.
            // Skip over the fixup so we keep that value.
            let skip_fixup = if !is_jump {
                Some(self.emit_unfilled_jmp())
            } else {
                None
            };

            let false_target = self.next_offset();
            self.emit(Instruction::LFALSESKIP { src: dst.0 });

            let true_target = self.next_offset();
            let true_idx = self.alloc_constant(Value::Boolean(true))?;
            self.emit(Instruction::LOAD {
                dst: dst.0,
                idx: true_idx,
            });

            let fl = std::mem::take(&mut expr.false_list);
            let tl = std::mem::take(&mut expr.true_list);
            self.patch_to(fl, false_target);
            self.patch_to(tl, true_target);

            if let Some(idx) = skip_fixup {
                let end = self.next_offset();
                let offset = end as i32 - (idx as i32 + 1);
                if let Instruction::JMP { offset: o } = &mut self.chunk.tape[idx] {
                    *o = offset;
                }
            }
        }

        expr.kind = ExprKind::Reg(dst);
        Ok(dst)
    }

    fn push_scope(&mut self) {
        self.scope.push(HashMap::new());
        self.scope_register_base.push(self.chunk.register_count);
        self.scope_close.push(Vec::new());
    }

    fn pop_scope(&mut self) -> Result<Vec<RegisterIndex>, CompileError> {
        self.scope.pop().ok_or_else(|| ice("missing scope"))?;
        self.chunk.register_count = self
            .scope_register_base
            .pop()
            .ok_or_else(|| ice("missing scope register base"))?;
        self.scope_close
            .pop()
            .ok_or_else(|| ice("missing scope close list"))
    }

    fn mark_close(&mut self, register: RegisterIndex) -> Result<(), CompileError> {
        self.scope_close
            .last_mut()
            .ok_or_else(|| ice("missing scope close list"))?
            .push(register);
        Ok(())
    }

    fn define(&mut self, name: String, data: VariableData) -> Result<(), CompileError> {
        let scope = self.scope.last_mut().ok_or_else(|| ice("missing scope"))?;
        scope.insert(name, data);
        Ok(())
    }

    fn resolve_local(&self, name: &str) -> Option<&VariableData> {
        for scope in self.scope.iter().rev() {
            if let Some(data) = scope.get(name) {
                return Some(data);
            }
        }
        None
    }

    fn alloc_constant(&mut self, value: Value<'gc>) -> Result<u16, CompileError> {
        // Check for an existing identical constant (exact type match, no int/float coercion).
        let existing = self.chunk.constants.iter().position(|c| match (c, &value) {
            (Value::Nil, Value::Nil) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
            (Value::String(a), Value::String(b)) => a == b,
            _ => false,
        });

        if let Some(idx) = existing {
            return Ok(idx as u16);
        }

        if self.chunk.constants.len() >= u16::MAX as usize {
            return Err(err(CompileErrorKind::Constants, LineNumber(0)));
        }
        let idx = self.chunk.constants.len() as u16;
        self.chunk.constants.push(value);
        Ok(idx)
    }

    fn alloc_string_constant(&mut self, s: &[u8]) -> Result<u16, CompileError> {
        let lua_str = LuaString::new(self.mc, s);
        self.alloc_constant(Value::String(lua_str))
    }

    /// Resolve a variable name as an upvalue from enclosing scopes.
    /// Deduplication is handled inside the capture callback (see `compile_nested`).
    fn resolve_upvalue(&mut self, name: &str) -> Option<u8> {
        (self.capture)(name)
    }
}

// ---------------------------------------------------------------------------
// Scope helpers
// ---------------------------------------------------------------------------

fn scope_lexical<F>(ctx: &mut Ctx, compile: F) -> Result<(), CompileError>
where
    F: FnOnce(&mut Ctx) -> Result<(), CompileError>,
{
    ctx.push_scope();
    compile(ctx)?;

    let close_regs = ctx.pop_scope()?;
    if let Some(first) = close_regs.first() {
        ctx.emit(Instruction::CLOSE { start: first.0 });
    }

    Ok(())
}

fn scope_break<F>(ctx: &mut Ctx, compile: F) -> Result<(), CompileError>
where
    F: FnOnce(&mut Ctx) -> Result<(), CompileError>,
{
    let label = ctx.new_label();
    ctx.control_end_label.push(label);
    compile(ctx)?;
    ctx.set_label(label, ctx.next_offset());
    ctx.control_end_label.pop();
    Ok(())
}

fn scope_lexical_break<F>(ctx: &mut Ctx, compile: F) -> Result<(), CompileError>
where
    F: FnOnce(&mut Ctx) -> Result<(), CompileError>,
{
    scope_lexical(ctx, |ctx| scope_break(ctx, compile))
}

// ---------------------------------------------------------------------------
// Top-level compilation entry point
// ---------------------------------------------------------------------------

pub fn compile<'gc>(
    mc: &Mutation<'gc>,
    root: &Root,
    interner: &TokenInterner,
) -> Result<Gc<'gc, Prototype<'gc>>, CompileError> {
    // The main chunk has one upvalue: _ENV (index 0).
    let mut no_parent_capture = |_name: &str| -> Option<u8> { None };

    let mut chunk = compile_function_to_chunk(
        mc,
        interner,
        &mut no_parent_capture,
        root.block(),
        std::iter::empty(),
        true, // main chunk is vararg
        0,
        None,
    )?;

    chunk.upvalue_desc = vec![UpValueDescriptor::ParentLocal(0)]; // _ENV
    Ok(chunk.assemble(mc))
}

/// Compile a function body into a Chunk (not yet assembled).
/// The caller is responsible for setting `chunk.upvalue_desc` and calling `chunk.assemble(mc)`.
#[allow(clippy::too_many_arguments)]
fn compile_function_to_chunk<'gc>(
    mc: &Mutation<'gc>,
    interner: &TokenInterner,
    parent_capture: &mut dyn FnMut(&str) -> Option<u8>,
    stmts: impl Iterator<Item = Stmt>,
    params: impl Iterator<Item = Ident>,
    is_vararg: bool,
    arity: u8,
    source: Option<LuaString<'gc>>,
) -> Result<Chunk<'gc>, CompileError> {
    let mut chunk = Chunk::new();
    chunk.is_vararg = is_vararg;
    chunk.arity = arity;
    chunk.source = source;

    let mut ctx = Ctx {
        interner,
        mc,
        chunk,
        control_end_label: Vec::new(),
        scope: Vec::new(),
        scope_register_base: Vec::new(),
        scope_close: Vec::new(),
        goto_labels: HashMap::new(),
        capture: Box::new(parent_capture),
        upvalue_desc: Vec::new(),
    };

    ctx.push_scope();

    // Emit VARARGPREP for vararg functions
    if is_vararg {
        ctx.emit(Instruction::VARARGPREP { num_fixed: arity });
    }

    // Allocate registers for parameters and bind them in scope
    for param in params {
        let name = param
            .name(interner)
            .ok_or_else(|| ice("parameter without name"))?;
        let reg = ctx.alloc_register()?;
        ctx.define(
            name.to_owned(),
            VariableData {
                register: reg,
                is_const: false,
            },
        )?;
    }

    // Compile the body
    for stmt in stmts {
        compile_stmt(&mut ctx, stmt)?;
    }

    // Emit implicit return at the end (skip if the last instruction is already a return)
    let needs_return = !matches!(ctx.chunk.tape.last(), Some(Instruction::RETURN { .. }));
    if needs_return {
        ctx.emit(Instruction::RETURN {
            values: 0,
            count: 1,
        });
    }

    let close_regs = ctx.pop_scope()?;
    if let Some(first) = close_regs.first() {
        // Insert CLOSE before the final RETURN
        let return_instr = ctx.chunk.tape.pop().unwrap();
        ctx.chunk.tape.push(Instruction::CLOSE { start: first.0 });
        ctx.chunk.tape.push(return_instr);
    }

    // The caller must set chunk.upvalue_desc before assembling.
    Ok(ctx.chunk)
}

// ---------------------------------------------------------------------------
// Statement compilation
// ---------------------------------------------------------------------------

fn compile_stmt(ctx: &mut Ctx, item: Stmt) -> Result<(), CompileError> {
    match item {
        Stmt::Label(item) => compile_label(ctx, item),
        Stmt::Goto(item) => compile_goto(ctx, item),
        Stmt::Decl(item) => compile_decl(ctx, item),
        Stmt::Assign(item) => compile_assign(ctx, item),
        Stmt::Func(item) => compile_func(ctx, item),
        Stmt::Expr(item) => {
            compile_expr_to_reg(ctx, item, None)?;
            Ok(())
        }
        Stmt::Break(item) => compile_break(ctx, item),
        Stmt::Return(item) => compile_return(ctx, item),
        Stmt::Do(item) => compile_do(ctx, item),
        Stmt::While(item) => compile_while(ctx, item),
        Stmt::Repeat(item) => compile_repeat(ctx, item),
        Stmt::If(item) => compile_if(ctx, item),
        Stmt::ForNum(item) => compile_for_num(ctx, item),
        Stmt::ForGen(item) => compile_for_gen(ctx, item),
    }
}

// ---------------------------------------------------------------------------
// Label / Goto
// ---------------------------------------------------------------------------

fn compile_label(ctx: &mut Ctx, item: Label) -> Result<(), CompileError> {
    let name = item
        .name()
        .ok_or_else(|| ice("label without name"))?
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?
        .to_owned();

    if let Some(&label_idx) = ctx.goto_labels.get(&name) {
        // Forward reference already allocated — resolve it now
        ctx.set_label(label_idx, ctx.next_offset());
    } else {
        let label_idx = ctx.new_label();
        ctx.set_label(label_idx, ctx.next_offset());
        ctx.goto_labels.insert(name, label_idx);
    }

    Ok(())
}

fn compile_goto(ctx: &mut Ctx, item: Goto) -> Result<(), CompileError> {
    let name = item
        .label()
        .ok_or_else(|| ice("goto without label"))?
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?
        .to_owned();

    if let Some(&label_idx) = ctx.goto_labels.get(&name) {
        ctx.emit_jump(label_idx);
    } else {
        // Forward goto — allocate a label that will be resolved when ::name:: is encountered
        let label_idx = ctx.new_label();
        ctx.goto_labels.insert(name, label_idx);
        ctx.emit_jump(label_idx);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

fn compile_decl(ctx: &mut Ctx, item: Decl) -> Result<(), CompileError> {
    // Check for `local function ...` syntax
    if let Some(func) = item.function() {
        let name = func
            .name(ctx.interner)
            .ok_or_else(|| ice("local function without name"))?
            .to_string();

        // Allocate register first so the function can reference itself
        let reg = ctx.alloc_register()?;
        ctx.define(
            name,
            VariableData {
                register: reg,
                is_const: false,
            },
        )?;

        let func_reg = compile_func_body(ctx, &func, Some(reg))?;
        if func_reg != reg {
            ctx.emit(Instruction::MOVE {
                dst: reg.0,
                src: func_reg.0,
            });
        }
        return Ok(());
    }

    let targets: Vec<_> = item
        .targets()
        .ok_or_else(|| ice("decl without targets"))?
        .collect();

    let values: Vec<_> = item
        .values()
        .map(|v| v.collect::<Vec<_>>())
        .unwrap_or_default();

    let num_targets = targets.len();
    let num_values = values.len();

    // Pre-allocate registers for all local variables
    let mut local_regs = Vec::with_capacity(num_targets);
    for _ in 0..num_targets {
        local_regs.push(ctx.alloc_register()?);
    }

    // Compile values into registers, using local register hints where possible
    let mut value_regs = Vec::new();
    for (i, expr) in values.into_iter().enumerate() {
        let is_last = i == num_values - 1;

        if is_last && num_targets > num_values {
            // Last value in a multi-target declaration — try to get multiple returns
            if let Expr::FuncCall(call) = expr {
                let want = num_targets - i;
                let regs = compile_expr_func_call(ctx, call, want)?;
                value_regs.extend(regs);
                continue;
            }
            if let Expr::VarArg = expr {
                let want = num_targets - i;
                let dst = ctx.alloc_register()?;
                ctx.emit(Instruction::VARARG {
                    dst: dst.0,
                    count: want as u8 + 1,
                });
                for j in 0..want {
                    value_regs.push(RegisterIndex(dst.0 + j as u8));
                }
                continue;
            }
        }

        // Pass the target local's register as a destination hint
        let hint = local_regs.get(i).copied();
        let reg = compile_expr_to_reg(ctx, expr, hint)?;
        value_regs.push(reg);
    }

    // Bind each target to its value (or nil)
    for (i, target) in targets.into_iter().enumerate() {
        let name = target
            .name()
            .ok_or_else(|| ice("decl target without name"))?
            .name(ctx.interner)
            .ok_or_else(|| ice("ident without name"))?
            .to_owned();

        let modifier = target.modifier();
        let is_const = matches!(modifier, Some(DeclModifier::Const));
        let is_close = matches!(modifier, Some(DeclModifier::Close));

        let local_reg = local_regs[i];
        let reg = if let Some(&val_reg) = value_regs.get(i) {
            if val_reg != local_reg {
                ctx.emit(Instruction::MOVE {
                    dst: local_reg.0,
                    src: val_reg.0,
                });
            }
            local_reg
        } else {
            // No value supplied — initialize to nil
            let idx = ctx.alloc_constant(Value::Nil)?;
            ctx.emit(Instruction::LOAD {
                dst: local_reg.0,
                idx,
            });
            local_reg
        };

        if is_close {
            ctx.emit(Instruction::TBC { val: reg.0 });
            ctx.mark_close(reg)?;
        }

        ctx.define(
            name,
            VariableData {
                register: reg,
                is_const,
            },
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

fn compile_assign(ctx: &mut Ctx, item: Assign) -> Result<(), CompileError> {
    let targets: Vec<_> = item
        .targets()
        .ok_or_else(|| ice("assign without targets"))?
        .collect();
    let values: Vec<_> = item
        .values()
        .ok_or_else(|| ice("assign without values"))?
        .collect();

    let num_targets = targets.len();
    let num_values = values.len();

    // Resolve destination hints for local variable targets
    let hints: Vec<Option<RegisterIndex>> = targets
        .iter()
        .map(|target| {
            if let Expr::Ident(ident) = target {
                let name = ident.name(ctx.interner)?;
                let data = ctx.resolve_local(name)?;
                if !data.is_const {
                    return Some(data.register);
                }
            }
            None
        })
        .collect();

    // Compile values
    let mut sources = Vec::new();
    for (i, expr) in values.into_iter().enumerate() {
        let is_last = i == num_values - 1;

        if is_last && num_targets > num_values {
            if let Expr::FuncCall(call) = expr {
                let want = num_targets - i;
                let regs = compile_expr_func_call(ctx, call, want)?;
                sources.extend(regs.into_iter().map(|r| r.0));
                continue;
            }
        }

        let hint = hints.get(i).copied().flatten();
        let reg = compile_expr_to_reg(ctx, expr, hint)?;
        sources.push(reg.0);
    }

    // Pad with nil if fewer values than targets
    while sources.len() < num_targets {
        let reg = ctx.alloc_register()?;
        let idx = ctx.alloc_constant(Value::Nil)?;
        ctx.emit(Instruction::LOAD { dst: reg.0, idx });
        sources.push(reg.0);
    }

    // Assign each target
    for (target, value) in targets.into_iter().zip(sources) {
        compile_assign_lhs(ctx, target, value)?;
    }

    Ok(())
}

fn compile_assign_lhs(ctx: &mut Ctx, target: Expr, value: u8) -> Result<(), CompileError> {
    match target {
        Expr::Ident(ident) => {
            let name = ident
                .name(ctx.interner)
                .ok_or_else(|| ice("ident without name"))?;

            // Local variable
            if let Some(data) = ctx.resolve_local(name) {
                if data.is_const {
                    return Err(err(
                        CompileErrorKind::Internal("assignment to const variable"),
                        LineNumber(0),
                    ));
                }
                let dst = data.register;
                if dst.0 != value {
                    ctx.emit(Instruction::MOVE {
                        dst: dst.0,
                        src: value,
                    });
                }
                return Ok(());
            }

            // Upvalue
            if let Some(idx) = ctx.resolve_upvalue(name) {
                ctx.emit(Instruction::SETUPVAL { src: value, idx });
                return Ok(());
            }

            // Global: _ENV[name]
            let key = ctx.alloc_string_constant(name.as_bytes())?;
            ctx.emit(Instruction::SETTABUP {
                src: value,
                idx: 0,
                key,
            });
            Ok(())
        }

        Expr::Index(index) => {
            let table_expr = index.target().ok_or_else(|| ice("index without target"))?;
            let key_expr = index.index().ok_or_else(|| ice("index without key"))?;
            let table = compile_expr_to_reg(ctx, table_expr, None)?;
            let key = compile_expr_to_reg(ctx, key_expr, None)?;
            ctx.emit(Instruction::SETTABLE {
                src: value,
                table: table.0,
                key: key.0,
            });
            Ok(())
        }

        Expr::BinaryOp(binop) => {
            // Property access: a.b = value
            let op = binop.op().ok_or_else(|| ice("binop without op"))?;
            if op == BinaryOperator::Property {
                let table_expr = binop.lhs().ok_or_else(|| ice("binop without lhs"))?;
                let field = binop.rhs().ok_or_else(|| ice("binop without rhs"))?;
                let table = compile_expr_to_reg(ctx, table_expr, None)?;
                let key = compile_property_key(ctx, field, None)?;
                ctx.emit(Instruction::SETTABLE {
                    src: value,
                    table: table.0,
                    key: key.0,
                });
                Ok(())
            } else {
                Err(ice("non-property binop as assignment target"))
            }
        }

        _ => Err(ice("invalid assignment target")),
    }
}

// ---------------------------------------------------------------------------
// Function statement
// ---------------------------------------------------------------------------

fn compile_func(ctx: &mut Ctx, item: Func) -> Result<(), CompileError> {
    // `function target() ... end` is sugar for `target = function() ... end`
    let target = item
        .target()
        .ok_or_else(|| ice("func stmt without target"))?;

    let func_reg = compile_func_body(ctx, &item, None)?;

    compile_assign_lhs(ctx, target, func_reg.0)?;

    Ok(())
}

fn compile_func_body(
    ctx: &mut Ctx,
    item: &Func,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let params: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
    let arity = params.len() as u8;

    let proto = compile_nested(ctx, stmts, params, false, arity)?;

    let proto_idx = ctx.chunk.prototypes.len() as u16;
    ctx.chunk.prototypes.push(proto);
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::CLOSURE {
        dst: dst.0,
        proto: proto_idx,
    });

    Ok(dst)
}

/// Compile a nested function, handling upvalue capture from the parent context.
fn compile_nested<'gc>(
    ctx: &mut Ctx<'gc, '_>,
    stmts: Vec<Stmt>,
    params: Vec<Ident>,
    is_vararg: bool,
    arity: u8,
) -> Result<Gc<'gc, Prototype<'gc>>, CompileError> {
    // Snapshot parent scope for the capture closure
    let scope_snapshot: Vec<HashMap<String, (u8, bool)>> = ctx
        .scope
        .iter()
        .map(|s| {
            s.iter()
                .map(|(k, v)| (k.clone(), (v.register.0, v.is_const)))
                .collect()
        })
        .collect();

    // Extract what we need from ctx before creating the closure,
    // to avoid borrowing all of ctx in the closure.
    let mc = ctx.mc;
    let interner = ctx.interner;
    let parent_capture = &mut *ctx.capture;

    let capture_list = RefCell::new(Vec::<(String, UpValueDescriptor)>::new());

    let mut capture_fn = |name: &str| -> Option<u8> {
        let mut list = capture_list.borrow_mut();

        // Already captured?
        for (i, (n, _)) in list.iter().enumerate() {
            if n == name {
                return Some(i as u8);
            }
        }

        // Search parent's local scopes
        for scope in scope_snapshot.iter().rev() {
            if let Some(&(reg, _)) = scope.get(name) {
                let idx = list.len() as u8;
                list.push((name.to_owned(), UpValueDescriptor::ParentLocal(reg)));
                return Some(idx);
            }
        }

        // Delegate to parent's upvalue resolution
        if let Some(parent_idx) = parent_capture(name) {
            let idx = list.len() as u8;
            list.push((
                name.to_owned(),
                UpValueDescriptor::ParentUpvalue(parent_idx),
            ));
            return Some(idx);
        }

        None
    };

    let mut chunk = compile_function_to_chunk(
        mc,
        interner,
        &mut capture_fn,
        stmts.into_iter(),
        params.into_iter(),
        is_vararg,
        arity,
        None,
    )?;

    // Inject the upvalue descriptors that were accumulated by capture_fn
    let list = capture_list.into_inner();
    chunk.upvalue_desc = list.into_iter().map(|(_, desc)| desc).collect();

    Ok(chunk.assemble(mc))
}

// ---------------------------------------------------------------------------
// Expression compilation
// ---------------------------------------------------------------------------

/// Compile an expression, returning an `ExprDesc` that may carry pending
/// jump lists instead of a materialised value (for comparisons, `not`, and
/// short-circuit `and`/`or`). Callers who need a concrete register call
/// `compile_expr_to_reg` instead, which discharges through the standard
/// `LFALSESKIP` / `LOAD true` fixup tail when lists are non-empty.
fn compile_expr(
    ctx: &mut Ctx,
    item: Expr,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    match item {
        Expr::PrefixOp(item) => compile_expr_prefix_op(ctx, item, dst),
        Expr::BinaryOp(item) => compile_expr_binary_op(ctx, item, dst),
        Expr::Method(item) => {
            let regs = compile_expr_method_call(ctx, item, 1)?;
            Ok(ExprDesc::from_reg(regs[0]))
        }
        Expr::Ident(item) => compile_expr_ident(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::Literal(item) => compile_expr_literal(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::Func(item) => compile_expr_func(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::Table(item) => compile_expr_table(ctx, item).map(ExprDesc::from_reg),
        Expr::FuncCall(item) => {
            let regs = compile_expr_func_call(ctx, item, 1)?;
            Ok(ExprDesc::from_reg(regs[0]))
        }
        Expr::Index(item) => compile_expr_index(ctx, item, dst).map(ExprDesc::from_reg),
        Expr::VarArg => {
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::VARARG {
                dst: dst.0,
                count: 2,
            });
            Ok(ExprDesc::from_reg(dst))
        }
    }
}

fn compile_expr_to_reg(
    ctx: &mut Ctx,
    item: Expr,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let mut expr = compile_expr(ctx, item, dst)?;
    ctx.discharge_to_reg_mut(&mut expr, dst)
}

fn compile_expr_ident(
    ctx: &mut Ctx,
    item: Ident,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let name = item
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?;

    // Local variable — returns its own register, ignores dst hint
    if let Some(data) = ctx.resolve_local(name) {
        return Ok(data.register);
    }

    // Upvalue
    if let Some(idx) = ctx.resolve_upvalue(name) {
        let dst = ctx.dst_or_alloc(dst)?;
        ctx.emit(Instruction::GETUPVAL { dst: dst.0, idx });
        return Ok(dst);
    }

    // Global: _ENV[name]
    let key = ctx.alloc_string_constant(name.as_bytes())?;
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::GETTABUP {
        dst: dst.0,
        idx: 0,
        key,
    });
    Ok(dst)
}

fn compile_expr_literal(
    ctx: &mut Ctx,
    item: Literal,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let value = item
        .value(ctx.interner)
        .ok_or_else(|| ice("literal without value"))?;

    let constant = match value {
        LiteralValue::Nil => Value::Nil,
        LiteralValue::Bool(b) => Value::Boolean(b),
        LiteralValue::Int(n) => Value::Integer(n),
        LiteralValue::Float(n) => Value::Float(n),
        LiteralValue::String(bytes) => Value::String(LuaString::new(ctx.mc, &bytes)),
    };

    let idx = ctx.alloc_constant(constant)?;
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::LOAD { dst: dst.0, idx });
    Ok(dst)
}

fn compile_expr_func(
    ctx: &mut Ctx,
    item: FuncExpr,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let params: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
    let arity = params.len() as u8;

    let proto = compile_nested(ctx, stmts, params, false, arity)?;

    let proto_idx = ctx.chunk.prototypes.len() as u16;
    ctx.chunk.prototypes.push(proto);
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::CLOSURE {
        dst: dst.0,
        proto: proto_idx,
    });

    Ok(dst)
}

fn compile_property_key(
    ctx: &mut Ctx,
    field: Expr,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    // Property access RHS should be an identifier used as a string key
    if let Expr::Ident(ident) = field {
        let name = ident
            .name(ctx.interner)
            .ok_or_else(|| ice("ident without name"))?;
        let idx = ctx.alloc_string_constant(name.as_bytes())?;
        let reg = ctx.dst_or_alloc(dst)?;
        ctx.emit(Instruction::LOAD { dst: reg.0, idx });
        Ok(reg)
    } else {
        // Fallback: compile as expression
        compile_expr_to_reg(ctx, field, dst)
    }
}

fn compile_expr_table(ctx: &mut Ctx, item: Table) -> Result<RegisterIndex, CompileError> {
    let dst = ctx.alloc_register()?;
    ctx.emit(Instruction::NEWTABLE { dst: dst.0 });

    let mut array_count: u16 = 0;
    let mut array_pending = 0u8;

    for entry in item.entries() {
        match entry {
            TableEntry::Array(arr) => {
                let value_expr = arr
                    .value()
                    .ok_or_else(|| ice("table array without value"))?;
                let val = compile_expr_to_reg(ctx, value_expr, None)?;

                // Place value in register after table for SETLIST
                let slot = ctx.alloc_register()?;
                if val != slot {
                    ctx.emit(Instruction::MOVE {
                        dst: slot.0,
                        src: val.0,
                    });
                }
                array_pending += 1;
                array_count += 1;

                // Flush when we hit the batch limit
                if array_pending >= 50 {
                    ctx.emit(Instruction::SETLIST {
                        table: dst.0,
                        count: array_pending,
                        offset: array_count - array_pending as u16,
                    });
                    array_pending = 0;
                }
            }

            TableEntry::Map(map) => {
                // Flush pending array entries first
                if array_pending > 0 {
                    ctx.emit(Instruction::SETLIST {
                        table: dst.0,
                        count: array_pending,
                        offset: array_count - array_pending as u16,
                    });
                    array_pending = 0;
                }

                let field = map.field().ok_or_else(|| ice("table map without field"))?;
                let field_name = field
                    .name(ctx.interner)
                    .ok_or_else(|| ice("ident without name"))?;
                let key_idx = ctx.alloc_string_constant(field_name.as_bytes())?;
                let key = ctx.alloc_register()?;
                ctx.emit(Instruction::LOAD {
                    dst: key.0,
                    idx: key_idx,
                });

                let value_expr = map.value().ok_or_else(|| ice("table map without value"))?;
                let val = compile_expr_to_reg(ctx, value_expr, None)?;

                ctx.emit(Instruction::SETTABLE {
                    src: val.0,
                    table: dst.0,
                    key: key.0,
                });
            }

            TableEntry::Generic(r#gen) => {
                // Flush pending array entries first
                if array_pending > 0 {
                    ctx.emit(Instruction::SETLIST {
                        table: dst.0,
                        count: array_pending,
                        offset: array_count - array_pending as u16,
                    });
                    array_pending = 0;
                }

                let key_expr = r#gen
                    .index()
                    .ok_or_else(|| ice("table generic without index"))?;
                let val_expr = r#gen
                    .value()
                    .ok_or_else(|| ice("table generic without value"))?;
                let key = compile_expr_to_reg(ctx, key_expr, None)?;
                let val = compile_expr_to_reg(ctx, val_expr, None)?;

                ctx.emit(Instruction::SETTABLE {
                    src: val.0,
                    table: dst.0,
                    key: key.0,
                });
            }
        }
    }

    // Flush remaining array entries
    if array_pending > 0 {
        ctx.emit(Instruction::SETLIST {
            table: dst.0,
            count: array_pending,
            offset: array_count - array_pending as u16,
        });
    }

    Ok(dst)
}

fn compile_expr_prefix_op(
    ctx: &mut Ctx,
    item: PrefixOp,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    let op = item.op().ok_or_else(|| ice("prefix op without operator"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("prefix op without operand"))?;

    // `not <expr>`: when the operand is a jump-list expression (comparison,
    // `not`-of-comparison, etc.) we swap its true/false lists and flip the
    // polarity of every pending control instruction — no NOT opcode, no
    // materialisation. For plain register/value operands fall back to the
    // NOT opcode so fall-through polarity stays consistent with the kind.
    if matches!(op, PrefixOperator::Not) {
        let mut inner = compile_expr(ctx, rhs_expr, None)?;
        match inner.kind {
            ExprKind::Jump => {
                ctx.negate_cond(&mut inner);
                return Ok(inner);
            }
            ExprKind::Reg(src) => {
                let dst = ctx.dst_or_alloc(dst)?;
                ctx.emit(Instruction::NOT {
                    dst: dst.0,
                    src: src.0,
                });
                return Ok(ExprDesc::from_reg(dst));
            }
        }
    }

    let src = compile_expr_to_reg(ctx, rhs_expr, None)?;

    let result_reg = match op {
        PrefixOperator::None => {
            // Unary + is a no-op
            src
        }
        PrefixOperator::Neg => {
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::UNM {
                dst: dst.0,
                src: src.0,
            });
            dst
        }
        PrefixOperator::Not => unreachable!("handled above"),
        PrefixOperator::Len => {
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::LEN {
                dst: dst.0,
                src: src.0,
            });
            dst
        }
        PrefixOperator::BitNot => {
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::BNOT {
                dst: dst.0,
                src: src.0,
            });
            dst
        }
    };
    Ok(ExprDesc::from_reg(result_reg))
}

fn compile_expr_binary_op(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<ExprDesc, CompileError> {
    let op = item.op().ok_or_else(|| ice("binary op without operator"))?;

    // Comparisons produce a jump-list expression so callers that only branch
    // on the result (if/while/repeat, `not`) can avoid the LOAD-true /
    // LFALSESKIP materialisation entirely. `and`/`or` still go through the
    // register-returning path for now — full jump-list treatment of those
    // comes in a later pass alongside TESTSET.
    match op {
        BinaryOperator::Eq
        | BinaryOperator::NEq
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::LEq
        | BinaryOperator::GEq => return compile_comparison_desc(ctx, item, op),
        _ => {}
    }

    compile_expr_binary_op_to_reg(ctx, item, dst).map(ExprDesc::from_reg)
}

fn compile_expr_binary_op_to_reg(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let op = item.op().ok_or_else(|| ice("binary op without operator"))?;

    // Short-circuit logical operators
    match op {
        BinaryOperator::And => return compile_logical_and(ctx, item, dst),
        BinaryOperator::Or => return compile_logical_or(ctx, item, dst),
        _ => {}
    }

    // Property access: a.b
    if op == BinaryOperator::Property {
        let lhs = item.lhs().ok_or_else(|| ice("binop without lhs"))?;
        let rhs = item.rhs().ok_or_else(|| ice("binop without rhs"))?;
        let table = compile_expr_to_reg(ctx, lhs, None)?;
        let key = compile_property_key(ctx, rhs, None)?;
        let dst = ctx.dst_or_alloc(dst)?;
        ctx.emit(Instruction::GETTABLE {
            dst: dst.0,
            table: table.0,
            key: key.0,
        });
        return Ok(dst);
    }

    // Method syntax in expressions (a:b) — should not appear here, handled by MethodCall
    if op == BinaryOperator::Method {
        return Err(ice("method operator in binary expression"));
    }

    let lhs_expr = item.lhs().ok_or_else(|| ice("binop without lhs"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("binop without rhs"))?;
    let lhs = compile_expr_to_reg(ctx, lhs_expr, None)?;
    let rhs = compile_expr_to_reg(ctx, rhs_expr, None)?;

    // Arithmetic and bitwise operations
    match op {
        BinaryOperator::Add => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::ADD {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Sub => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::SUB {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Mul => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::MUL {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Div => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::DIV {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::IntDiv => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::IDIV {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Mod => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::MOD {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Exp => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::POW {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::BitAnd => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::BAND {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::BitOr => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::BOR {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::BitXor => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::BXOR {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::LShift => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::SHL {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::RShift => emit_arith(
            ctx,
            lhs,
            rhs,
            Instruction::SHR {
                dst: 0,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst,
        ),
        BinaryOperator::Concat => {
            let dst = ctx.dst_or_alloc(dst)?;
            ctx.emit(Instruction::CONCAT {
                dst: dst.0,
                lhs: lhs.0,
                rhs: rhs.0,
            });
            Ok(dst)
        }

        BinaryOperator::Eq
        | BinaryOperator::NEq
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::LEq
        | BinaryOperator::GEq => unreachable!("comparisons handled by compile_comparison_desc"),

        BinaryOperator::And | BinaryOperator::Or => unreachable!("handled above"),
        BinaryOperator::Property | BinaryOperator::Method => unreachable!("handled above"),
    }
}

fn emit_arith(
    ctx: &mut Ctx,
    lhs: RegisterIndex,
    rhs: RegisterIndex,
    mut instr: Instruction,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let dst = ctx.dst_or_alloc(dst)?;
    // Patch the dst field in the instruction
    match &mut instr {
        Instruction::ADD { dst: d, .. }
        | Instruction::SUB { dst: d, .. }
        | Instruction::MUL { dst: d, .. }
        | Instruction::DIV { dst: d, .. }
        | Instruction::IDIV { dst: d, .. }
        | Instruction::MOD { dst: d, .. }
        | Instruction::POW { dst: d, .. }
        | Instruction::BAND { dst: d, .. }
        | Instruction::BOR { dst: d, .. }
        | Instruction::BXOR { dst: d, .. }
        | Instruction::SHL { dst: d, .. }
        | Instruction::SHR { dst: d, .. } => *d = dst.0,
        _ => unreachable!(),
    }
    ctx.emit(instr);
    Ok(dst)
}

/// Compile a comparison (`==`, `~=`, `<`, `>`, `<=`, `>=`) as a jump-list
/// expression: emit just `CMP` + an unfilled `JMP` and return an
/// `ExprDesc::Jump` whose false-list holds the JMP. Consumers that branch
/// on the result patch the list directly (2 instructions); consumers that
/// need a boolean in a register call `discharge_to_reg_mut`, which emits
/// the standard `LFALSESKIP` / `LOAD K(true)` fixup tail.
fn compile_comparison_desc(
    ctx: &mut Ctx,
    item: BinaryOp,
    op: BinaryOperator,
) -> Result<ExprDesc, CompileError> {
    let lhs_expr = item.lhs().ok_or_else(|| ice("cmp without lhs"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("cmp without rhs"))?;
    let lhs = compile_expr_to_reg(ctx, lhs_expr, None)?;
    let rhs = compile_expr_to_reg(ctx, rhs_expr, None)?;

    // `inverted = false` on all comparison ops: the VM's `op_cmp` skips the
    // following instruction iff `(cmp_result != inverted) == true`, i.e.
    // skip when the comparison succeeds. So the unfilled JMP fires when the
    // comparison FAILS, which is exactly a "jump on false" — append into
    // `false_list`.
    let instr = match op {
        BinaryOperator::Eq => Instruction::EQ {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: false,
        },
        BinaryOperator::NEq => Instruction::EQ {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: true,
        },
        BinaryOperator::Lt => Instruction::LT {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: false,
        },
        BinaryOperator::Gt => Instruction::LT {
            lhs: rhs.0,
            rhs: lhs.0,
            inverted: false,
        },
        BinaryOperator::LEq => Instruction::LE {
            lhs: lhs.0,
            rhs: rhs.0,
            inverted: false,
        },
        BinaryOperator::GEq => Instruction::LE {
            lhs: rhs.0,
            rhs: lhs.0,
            inverted: false,
        },
        _ => return Err(ice("compile_comparison_desc called with non-comparison op")),
    };
    ctx.emit(instr);
    let jmp = ctx.emit_unfilled_jmp();

    Ok(ExprDesc {
        kind: ExprKind::Jump,
        true_list: JumpList::new(),
        false_list: JumpList::single(jmp),
    })
}

/// Compile `lhs and rhs` / `lhs or rhs` with TESTSET-based short-circuiting
/// so the result preserves the operand that decided the short-circuit.
///
/// `and`: if `lhs` is falsy, the whole expression is `lhs` — skip `rhs` and
/// keep `lhs` in `dst`. If `lhs` is truthy, evaluate `rhs` (which overwrites
/// `dst`) and its value is the result.
///
/// `or`: symmetric. Truthy `lhs` short-circuits with `lhs` as the result.
///
/// The TESTSET does the "assign `lhs` into `dst` on the short-circuit leg"
/// work in one instruction, replacing the old TEST + MOVE pattern.
fn compile_logical_short_circuit(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
    is_and: bool,
) -> Result<RegisterIndex, CompileError> {
    let kw = if is_and { "and" } else { "or" };
    let lhs_expr = item.lhs().ok_or_else(|| ice("short-circuit without lhs"))?;
    let rhs_expr = item.rhs().ok_or_else(|| {
        ice(if is_and {
            "and without rhs"
        } else {
            "or without rhs"
        })
    })?;
    let _ = kw; // kw only used for error text in ice() above; silence unused warning

    let dst = ctx.dst_or_alloc(dst)?;
    let lhs = compile_expr_to_reg(ctx, lhs_expr, Some(dst))?;

    // TESTSET semantics (our impl): `if truthy(src) == inverted then skip; else R[dst] := R[src]`.
    //
    // `and` (is_and=true): short-circuit when lhs is FALSY, result = lhs.
    //   Use inverted=true → truthy path skips next (JMP), no assign needed
    //   because truthy means we're going to evaluate rhs and overwrite dst.
    //   Falsy path assigns dst := lhs and falls through to the JMP → end.
    // `or`: short-circuit when lhs is TRUTHY, result = lhs.
    //   Use inverted=false → falsy path skips next (JMP), truthy path assigns
    //   dst := lhs and falls through to the JMP → end.
    let end_label = ctx.new_label();
    ctx.emit(Instruction::TESTSET {
        dst: dst.0,
        src: lhs.0,
        inverted: is_and,
    });
    ctx.emit_jump(end_label);

    let rhs = compile_expr_to_reg(ctx, rhs_expr, Some(dst))?;
    if rhs != dst {
        ctx.emit(Instruction::MOVE {
            dst: dst.0,
            src: rhs.0,
        });
    }

    ctx.set_label(end_label, ctx.next_offset());
    Ok(dst)
}

fn compile_logical_and(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    compile_logical_short_circuit(ctx, item, dst, true)
}

fn compile_logical_or(
    ctx: &mut Ctx,
    item: BinaryOp,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    compile_logical_short_circuit(ctx, item, dst, false)
}

fn compile_expr_func_call(
    ctx: &mut Ctx,
    item: FuncCall,
    want: usize,
) -> Result<Vec<RegisterIndex>, CompileError> {
    let target = item
        .target()
        .ok_or_else(|| ice("func call without target"))?;
    let func = compile_expr_to_reg(ctx, target, None)?;

    let args: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let nargs = args.len();

    // Compile arguments into consecutive registers after func
    for (i, arg_expr) in args.into_iter().enumerate() {
        let expected_reg = RegisterIndex(func.0 + 1 + i as u8);
        let arg = compile_expr_to_reg(ctx, arg_expr, Some(expected_reg))?;
        if arg != expected_reg {
            // Need to ensure we have the register allocated
            while ctx.chunk.register_count <= expected_reg.0 as usize {
                ctx.alloc_register()?;
            }
            ctx.emit(Instruction::MOVE {
                dst: expected_reg.0,
                src: arg.0,
            });
        }
    }

    ctx.emit(Instruction::CALL {
        func: func.0,
        args: nargs as u8 + 1,
        returns: want as u8 + 1,
    });

    // Results are placed starting at func register
    let mut results = Vec::with_capacity(want);
    for i in 0..want {
        results.push(RegisterIndex(func.0 + i as u8));
    }
    Ok(results)
}

fn compile_expr_method_call(
    ctx: &mut Ctx,
    item: MethodCall,
    want: usize,
) -> Result<Vec<RegisterIndex>, CompileError> {
    let object_expr = item
        .object()
        .ok_or_else(|| ice("method call without object"))?;
    let method_ident = item
        .method()
        .ok_or_else(|| ice("method call without method name"))?;
    let method_name = method_ident
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?;

    let object = compile_expr_to_reg(ctx, object_expr, None)?;

    // Load method name as key
    let key_idx = ctx.alloc_string_constant(method_name.as_bytes())?;
    let key_reg = ctx.alloc_register()?;
    ctx.emit(Instruction::LOAD {
        dst: key_reg.0,
        idx: key_idx,
    });

    // Get the method function: R[func] = R[object][key]
    let func = ctx.alloc_register()?;
    ctx.emit(Instruction::GETTABLE {
        dst: func.0,
        table: object.0,
        key: key_reg.0,
    });

    // First argument is self (the object)
    let self_reg = RegisterIndex(func.0 + 1);
    while ctx.chunk.register_count <= self_reg.0 as usize {
        ctx.alloc_register()?;
    }
    ctx.emit(Instruction::MOVE {
        dst: self_reg.0,
        src: object.0,
    });

    // Compile remaining arguments
    let args: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let nargs = args.len() + 1; // +1 for self

    for (i, arg_expr) in args.into_iter().enumerate() {
        let expected_reg = RegisterIndex(func.0 + 2 + i as u8);
        let arg = compile_expr_to_reg(ctx, arg_expr, Some(expected_reg))?;
        if arg != expected_reg {
            while ctx.chunk.register_count <= expected_reg.0 as usize {
                ctx.alloc_register()?;
            }
            ctx.emit(Instruction::MOVE {
                dst: expected_reg.0,
                src: arg.0,
            });
        }
    }

    ctx.emit(Instruction::CALL {
        func: func.0,
        args: nargs as u8 + 1,
        returns: want as u8 + 1,
    });

    let mut results = Vec::with_capacity(want);
    for i in 0..want {
        results.push(RegisterIndex(func.0 + i as u8));
    }
    Ok(results)
}

fn compile_expr_index(
    ctx: &mut Ctx,
    item: Index,
    dst: Option<RegisterIndex>,
) -> Result<RegisterIndex, CompileError> {
    let target_expr = item.target().ok_or_else(|| ice("index without target"))?;
    let key_expr = item.index().ok_or_else(|| ice("index without key"))?;
    let table = compile_expr_to_reg(ctx, target_expr, None)?;
    let key = compile_expr_to_reg(ctx, key_expr, None)?;
    let dst = ctx.dst_or_alloc(dst)?;
    ctx.emit(Instruction::GETTABLE {
        dst: dst.0,
        table: table.0,
        key: key.0,
    });
    Ok(dst)
}

// ---------------------------------------------------------------------------
// Break / Return
// ---------------------------------------------------------------------------

fn compile_break(ctx: &mut Ctx, _item: Break) -> Result<(), CompileError> {
    let label = *ctx
        .control_end_label
        .last()
        .ok_or_else(|| ice("break outside of loop"))?;
    ctx.emit_jump(label);
    Ok(())
}

fn compile_return(ctx: &mut Ctx, item: Return) -> Result<(), CompileError> {
    let exprs: Vec<_> = item.exprs().map(|e| e.collect()).unwrap_or_default();

    if exprs.is_empty() {
        ctx.emit(Instruction::RETURN {
            values: 0,
            count: 1,
        });
        return Ok(());
    }

    let n = exprs.len();
    let first_reg = ctx.alloc_register()?;

    for (i, expr) in exprs.into_iter().enumerate() {
        let target = RegisterIndex(first_reg.0 + i as u8);
        let reg = compile_expr_to_reg(ctx, expr, Some(target))?;
        if reg != target {
            while ctx.chunk.register_count <= target.0 as usize {
                ctx.alloc_register()?;
            }
            ctx.emit(Instruction::MOVE {
                dst: target.0,
                src: reg.0,
            });
        }
    }

    ctx.emit(Instruction::RETURN {
        values: first_reg.0,
        count: n as u8 + 1,
    });

    Ok(())
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

fn compile_do(ctx: &mut Ctx, item: Do) -> Result<(), CompileError> {
    scope_lexical(ctx, |ctx| {
        let stmts: Vec<_> = item.stmts().map(|s| s.collect()).unwrap_or_default();
        for stmt in stmts {
            compile_stmt(ctx, stmt)?;
        }
        Ok(())
    })
}

/// Compile `cond` as the condition of a branching construct. Returns the
/// list of jumps that should be patched to the branch-out target (i.e. the
/// "else" of an `if`, the "break" of a `while`, the loop-back of a
/// `repeat ... until`). The fall-through after this call is the path taken
/// when the condition evaluates to truthy.
fn compile_branch_cond_false(ctx: &mut Ctx, cond_expr: Expr) -> Result<JumpList, CompileError> {
    let mut desc = compile_expr(ctx, cond_expr, None)?;
    ctx.goiffalse(&mut desc)?;
    // After goiffalse, `false_list` holds every jump that fires on falsy
    // (the caller patches these to the branch-out target) and `true_list`
    // holds any jumps accumulated during LHS evaluation of an `and`/`or`
    // chain that need to land here at the truthy-fall-through position.
    let true_list = std::mem::take(&mut desc.true_list);
    ctx.patch_to_here(true_list);
    Ok(desc.false_list)
}

fn compile_while(ctx: &mut Ctx, item: While) -> Result<(), CompileError> {
    scope_lexical_break(ctx, |ctx| {
        let loop_start = ctx.new_label();
        ctx.set_label(loop_start, ctx.next_offset());

        let cond_expr = item.cond().ok_or_else(|| ice("while without condition"))?;
        let break_list = compile_branch_cond_false(ctx, cond_expr)?;

        // Compile body
        if let Some(block) = item.block() {
            let stmts: Vec<_> = block.stmts().map(|s| s.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }
        }

        // Jump back to condition check
        ctx.emit_jump(loop_start);

        // Patch all "condition false" jumps to the post-loop break target.
        let break_label = *ctx
            .control_end_label
            .last()
            .ok_or_else(|| ice("missing break label"))?;
        let break_target_idx = break_label as usize;
        for idx in break_list.jumps {
            ctx.chunk.jump_patches.push((idx, break_target_idx as u16));
        }

        Ok(())
    })
}

fn compile_repeat(ctx: &mut Ctx, item: Repeat) -> Result<(), CompileError> {
    scope_break(ctx, |ctx| {
        // Note: repeat-until has special scoping — the condition can see
        // locals defined in the body. So we wrap the body AND condition
        // in the same lexical scope.
        scope_lexical(ctx, |ctx| {
            let loop_start_off = ctx.next_offset();

            // Compile body
            let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }

            // Compile condition — jumps from false_list (condition falsy)
            // get patched back to the loop start. Truthy fall-through exits
            // the loop.
            let cond_expr = item.cond().ok_or_else(|| ice("repeat without condition"))?;
            let false_list = compile_branch_cond_false(ctx, cond_expr)?;
            ctx.patch_to(false_list, loop_start_off);

            Ok(())
        })
    })
}

fn compile_if(ctx: &mut Ctx, item: If) -> Result<(), CompileError> {
    let end_label = ctx.new_label();

    compile_if_chain(ctx, item, end_label)?;

    ctx.set_label(end_label, ctx.next_offset());
    Ok(())
}

fn compile_if_chain(ctx: &mut Ctx, item: If, end_label: u16) -> Result<(), CompileError> {
    let cond_expr = item.cond().ok_or_else(|| ice("if without condition"))?;
    let else_list = compile_branch_cond_false(ctx, cond_expr)?;

    // Then block
    scope_lexical(ctx, |ctx| {
        let stmts: Vec<_> = item.stmts().map(|s| s.collect()).unwrap_or_default();
        for stmt in stmts {
            compile_stmt(ctx, stmt)?;
        }
        Ok(())
    })?;

    // Only emit jump-to-end when there's an else/elseif (otherwise we just fall through)
    if item.else_chain().is_some() {
        ctx.emit_jump(end_label);
    }

    // Patch the "condition false" jumps to this point (start of else branch).
    ctx.patch_to_here(else_list);

    // Else chain
    if let Some(chain) = item.else_chain() {
        if let Some(elseif) = chain.elseif_block() {
            compile_if_chain(ctx, elseif, end_label)?;
        } else if let Some(else_stmts) = chain.else_block() {
            scope_lexical(ctx, |ctx| {
                let stmts: Vec<_> = else_stmts.collect();
                for stmt in stmts {
                    compile_stmt(ctx, stmt)?;
                }
                Ok(())
            })?;
        }
    }

    Ok(())
}

fn compile_for_num(ctx: &mut Ctx, item: ForNum) -> Result<(), CompileError> {
    scope_lexical_break(ctx, |ctx| {
        let (counter_ident, init_expr) = item
            .counter()
            .ok_or_else(|| ice("for_num without counter"))?;
        let counter_name = counter_ident
            .name(ctx.interner)
            .ok_or_else(|| ice("ident without name"))?
            .to_owned();

        let limit_expr = item.end().ok_or_else(|| ice("for_num without limit"))?;

        // Pre-allocate consecutive registers: base, base+1, base+2
        let base = ctx.alloc_register()?;
        let limit_reg = ctx.alloc_register()?;
        let step_reg = ctx.alloc_register()?;

        // Compile init, limit, step with destination hints
        let init = compile_expr_to_reg(ctx, init_expr, Some(base))?;
        if init != base {
            ctx.emit(Instruction::MOVE {
                dst: base.0,
                src: init.0,
            });
        }

        let limit = compile_expr_to_reg(ctx, limit_expr, Some(limit_reg))?;
        if limit != limit_reg {
            ctx.emit(Instruction::MOVE {
                dst: limit_reg.0,
                src: limit.0,
            });
        }

        if let Some(step_expr) = item.step() {
            let step = compile_expr_to_reg(ctx, step_expr, Some(step_reg))?;
            if step != step_reg {
                ctx.emit(Instruction::MOVE {
                    dst: step_reg.0,
                    src: step.0,
                });
            }
        } else {
            // Default step = 1
            let one_idx = ctx.alloc_constant(Value::Integer(1))?;
            ctx.emit(Instruction::LOAD {
                dst: step_reg.0,
                idx: one_idx,
            });
        }

        // base+3 is the visible loop variable
        let loop_var = ctx.alloc_register()?;
        ctx.define(
            counter_name,
            VariableData {
                register: loop_var,
                is_const: false,
            },
        )?;

        let loop_body = ctx.new_label();
        let loop_end = ctx.new_label();

        // FORPREP: initialize and jump past body if loop shouldn't execute
        ctx.emit_jump_instr(
            loop_end,
            Instruction::FORPREP {
                base: base.0,
                offset: 0,
            },
        );

        ctx.set_label(loop_body, ctx.next_offset());

        // Body
        if let Some(block) = item.block() {
            let stmts: Vec<_> = block.stmts().map(|s| s.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }
        }

        // FORLOOP: increment and jump back if still in range
        ctx.emit_jump_instr(
            loop_body,
            Instruction::FORLOOP {
                base: base.0,
                offset: 0,
            },
        );

        ctx.set_label(loop_end, ctx.next_offset());
        Ok(())
    })
}

fn compile_for_gen(ctx: &mut Ctx, item: ForGen) -> Result<(), CompileError> {
    scope_lexical_break(ctx, |ctx| {
        // Compile iterator expressions into base, base+1, base+2
        let values: Vec<_> = item
            .values()
            .ok_or_else(|| ice("for_gen without values"))?
            .collect();
        let targets: Vec<_> = item
            .targets()
            .ok_or_else(|| ice("for_gen without targets"))?
            .collect();
        let num_targets = targets.len();

        // We need 3 control registers + N target registers
        // R[base] = iterator function
        // R[base+1] = state
        // R[base+2] = initial control variable
        // R[base+3]..R[base+2+N] = loop variables
        let base = ctx.alloc_register()?;
        // Pre-allocate the remaining two control registers
        ctx.alloc_register()?; // base+1
        ctx.alloc_register()?; // base+2

        // Compile up to 3 iterator values with destination hints
        for (i, val_expr) in values.into_iter().enumerate().take(3) {
            let target_reg = RegisterIndex(base.0 + i as u8);
            let val = compile_expr_to_reg(ctx, val_expr, Some(target_reg))?;
            if val != target_reg {
                while ctx.chunk.register_count <= target_reg.0 as usize {
                    ctx.alloc_register()?;
                }
                ctx.emit(Instruction::MOVE {
                    dst: target_reg.0,
                    src: val.0,
                });
            }
        }

        // Allocate registers for loop variables and bind them
        for target_ident in targets {
            let name = target_ident
                .name(ctx.interner)
                .ok_or_else(|| ice("ident without name"))?
                .to_owned();
            let reg = ctx.alloc_register()?;
            ctx.define(
                name,
                VariableData {
                    register: reg,
                    is_const: false,
                },
            )?;
        }

        let loop_body = ctx.new_label();
        let loop_test = ctx.new_label();

        // TFORPREP: jump to the test
        ctx.emit_jump_instr(
            loop_test,
            Instruction::TFORPREP {
                base: base.0,
                offset: 0,
            },
        );

        ctx.set_label(loop_body, ctx.next_offset());

        // Body
        if let Some(block) = item.block() {
            let stmts: Vec<_> = block.stmts().map(|s| s.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }
        }

        ctx.set_label(loop_test, ctx.next_offset());

        // TFORCALL: call iterator, results go to base+3..base+2+count
        ctx.emit(Instruction::TFORCALL {
            base: base.0,
            count: num_targets as u8,
        });

        // TFORLOOP: if control variable is not nil, jump back to body
        ctx.emit_jump_instr(
            loop_body,
            Instruction::TFORLOOP {
                base: base.0,
                offset: 0,
            },
        );

        Ok(())
    })
}
