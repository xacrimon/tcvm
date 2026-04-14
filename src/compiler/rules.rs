use std::collections::HashMap;

use cstree::interning::TokenInterner;

use super::defs::{Chunk, RegisterIndex};
use super::{CompileError, CompileErrorKind, LineNumber};
use crate::dmm::{Gc, Mutation};
use crate::env::{LuaString, Prototype, Value};
use crate::instruction::{Instruction, MetaMethod, UpValueDescriptor};
use crate::parser::syntax::{
    Assign, BinaryOp, BinaryOperator, Break, Decl, DeclModifier, Do, Expr, ForGen, ForNum, Func,
    FuncCall, FuncExpr, Goto, Ident, If, Index, Label, Literal, LiteralValue, MethodCall,
    PrefixOp, PrefixOperator, Repeat, Return, Root, Stmt, Table, TableEntry, While,
};

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
        let scope = self
            .scope
            .last_mut()
            .ok_or_else(|| ice("missing scope"))?;
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

    fn resolve_upvalue(&mut self, name: &str) -> Option<u8> {
        // Check if we already captured this name
        for (i, desc) in self.upvalue_desc.iter().enumerate() {
            // We need to match by name, so we track names separately
            // Actually, upvalue_desc doesn't store names. We need a side map.
            // For now, delegate to the capture callback which handles dedup.
            let _ = (i, desc);
        }
        (self.capture)(name)
    }
}

// ---------------------------------------------------------------------------
// Upvalue capture helper
// ---------------------------------------------------------------------------

/// Resolve a variable name as an upvalue by searching enclosing scopes.
fn make_capture_fn<'gc, 'a>(
    parent_scope: &'a [HashMap<String, VariableData>],
    parent_capture: &'a mut dyn FnMut(&str) -> Option<u8>,
    captured: &'a mut Vec<(String, UpValueDescriptor)>,
) -> impl FnMut(&str) -> Option<u8> + 'a {
    move |name: &str| -> Option<u8> {
        // Already captured?
        for (i, (n, _)) in captured.iter().enumerate() {
            if n == name {
                return Some(i as u8);
            }
        }

        // Search parent's local scopes
        for scope in parent_scope.iter().rev() {
            if let Some(data) = scope.get(name) {
                let idx = captured.len() as u8;
                captured.push((
                    name.to_owned(),
                    UpValueDescriptor::ParentLocal(data.register.0),
                ));
                return Some(idx);
            }
        }

        // Delegate to parent's capture (it's an upvalue of the parent too)
        if let Some(parent_idx) = parent_capture(name) {
            let idx = captured.len() as u8;
            captured.push((
                name.to_owned(),
                UpValueDescriptor::ParentUpvalue(parent_idx),
            ));
            return Some(idx);
        }

        None
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
        ctx.chunk
            .tape
            .push(Instruction::CLOSE { start: first.0 });
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
            compile_expr(ctx, item)?;
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

        let func_reg = compile_func_body(ctx, &func)?;
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

    // Compile values into registers
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

        let reg = compile_expr(ctx, expr)?;
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

        let reg = if let Some(&val_reg) = value_regs.get(i) {
            // If the value is already in a fresh register we can reuse it;
            // otherwise move to a new register for the local variable.
            let local_reg = ctx.alloc_register()?;
            if val_reg != local_reg {
                ctx.emit(Instruction::MOVE {
                    dst: local_reg.0,
                    src: val_reg.0,
                });
            }
            local_reg
        } else {
            // No value supplied — initialize to nil
            let local_reg = ctx.alloc_register()?;
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

        ctx.define(name, VariableData { register: reg, is_const })?;
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

        let reg = compile_expr(ctx, expr)?;
        sources.push(reg.0);
    }

    // Pad with nil if fewer values than targets
    while sources.len() < num_targets {
        let reg = ctx.alloc_register()?;
        let idx = ctx.alloc_constant(Value::Nil)?;
        ctx.emit(Instruction::LOAD {
            dst: reg.0,
            idx,
        });
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
            let table_expr = index
                .target()
                .ok_or_else(|| ice("index without target"))?;
            let key_expr = index
                .index()
                .ok_or_else(|| ice("index without key"))?;
            let table = compile_expr(ctx, table_expr)?;
            let key = compile_expr(ctx, key_expr)?;
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
                let table = compile_expr(ctx, table_expr)?;
                let key = compile_property_key(ctx, field)?;
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

    let func_reg = compile_func_body(ctx, &item)?;

    compile_assign_lhs(ctx, target, func_reg.0)?;

    Ok(())
}

fn compile_func_body(ctx: &mut Ctx, item: &Func) -> Result<RegisterIndex, CompileError> {
    let params: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
    let arity = params.len() as u8;

    let proto = compile_nested(ctx, stmts, params, false, arity)?;

    let proto_idx = ctx.chunk.prototypes.len() as u16;
    ctx.chunk.prototypes.push(proto);
    let dst = ctx.alloc_register()?;
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

    let capture_list = std::cell::RefCell::new(Vec::<(String, UpValueDescriptor)>::new());

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

fn compile_expr(ctx: &mut Ctx, item: Expr) -> Result<RegisterIndex, CompileError> {
    match item {
        Expr::Method(item) => {
            let regs = compile_expr_method_call(ctx, item, 1)?;
            Ok(regs[0])
        }
        Expr::Ident(item) => compile_expr_ident(ctx, item),
        Expr::Literal(item) => compile_expr_literal(ctx, item),
        Expr::Func(item) => compile_expr_func(ctx, item),
        Expr::Table(item) => compile_expr_table(ctx, item),
        Expr::PrefixOp(item) => compile_expr_prefix_op(ctx, item),
        Expr::BinaryOp(item) => compile_expr_binary_op(ctx, item),
        Expr::FuncCall(item) => {
            let regs = compile_expr_func_call(ctx, item, 1)?;
            Ok(regs[0])
        }
        Expr::Index(item) => compile_expr_index(ctx, item),
        Expr::VarArg => {
            let dst = ctx.alloc_register()?;
            ctx.emit(Instruction::VARARG { dst: dst.0, count: 2 });
            Ok(dst)
        }
    }
}

fn compile_expr_ident(ctx: &mut Ctx, item: Ident) -> Result<RegisterIndex, CompileError> {
    let name = item
        .name(ctx.interner)
        .ok_or_else(|| ice("ident without name"))?;

    // Local variable
    if let Some(data) = ctx.resolve_local(name) {
        return Ok(data.register);
    }

    // Upvalue
    if let Some(idx) = ctx.resolve_upvalue(name) {
        let dst = ctx.alloc_register()?;
        ctx.emit(Instruction::GETUPVAL { dst: dst.0, idx });
        return Ok(dst);
    }

    // Global: _ENV[name]
    let key = ctx.alloc_string_constant(name.as_bytes())?;
    let dst = ctx.alloc_register()?;
    ctx.emit(Instruction::GETTABUP {
        dst: dst.0,
        idx: 0,
        key,
    });
    Ok(dst)
}

fn compile_expr_literal(ctx: &mut Ctx, item: Literal) -> Result<RegisterIndex, CompileError> {
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
    let dst = ctx.alloc_register()?;
    ctx.emit(Instruction::LOAD { dst: dst.0, idx });
    Ok(dst)
}

fn compile_expr_func(ctx: &mut Ctx, item: FuncExpr) -> Result<RegisterIndex, CompileError> {
    let params: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
    let arity = params.len() as u8;

    let proto = compile_nested(ctx, stmts, params, false, arity)?;

    let proto_idx = ctx.chunk.prototypes.len() as u16;
    ctx.chunk.prototypes.push(proto);
    let dst = ctx.alloc_register()?;
    ctx.emit(Instruction::CLOSURE {
        dst: dst.0,
        proto: proto_idx,
    });

    Ok(dst)
}

fn compile_property_key(ctx: &mut Ctx, field: Expr) -> Result<RegisterIndex, CompileError> {
    // Property access RHS should be an identifier used as a string key
    if let Expr::Ident(ident) = field {
        let name = ident
            .name(ctx.interner)
            .ok_or_else(|| ice("ident without name"))?;
        let idx = ctx.alloc_string_constant(name.as_bytes())?;
        let reg = ctx.alloc_register()?;
        ctx.emit(Instruction::LOAD { dst: reg.0, idx });
        Ok(reg)
    } else {
        // Fallback: compile as expression
        compile_expr(ctx, field)
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
                let value_expr = arr.value().ok_or_else(|| ice("table array without value"))?;
                let val = compile_expr(ctx, value_expr)?;

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
                let val = compile_expr(ctx, value_expr)?;

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
                let key = compile_expr(ctx, key_expr)?;
                let val = compile_expr(ctx, val_expr)?;

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

fn compile_expr_prefix_op(ctx: &mut Ctx, item: PrefixOp) -> Result<RegisterIndex, CompileError> {
    let op = item.op().ok_or_else(|| ice("prefix op without operator"))?;
    let rhs_expr = item.rhs().ok_or_else(|| ice("prefix op without operand"))?;
    let src = compile_expr(ctx, rhs_expr)?;

    match op {
        PrefixOperator::None => {
            // Unary + is a no-op
            Ok(src)
        }
        PrefixOperator::Neg => {
            let dst = ctx.alloc_register()?;
            ctx.emit(Instruction::UNM {
                dst: dst.0,
                src: src.0,
            });
            Ok(dst)
        }
        PrefixOperator::Not => {
            let dst = ctx.alloc_register()?;
            ctx.emit(Instruction::NOT {
                dst: dst.0,
                src: src.0,
            });
            Ok(dst)
        }
        PrefixOperator::Len => {
            let dst = ctx.alloc_register()?;
            ctx.emit(Instruction::LEN {
                dst: dst.0,
                src: src.0,
            });
            Ok(dst)
        }
        PrefixOperator::BitNot => {
            let dst = ctx.alloc_register()?;
            ctx.emit(Instruction::BNOT {
                dst: dst.0,
                src: src.0,
            });
            Ok(dst)
        }
    }
}

fn compile_expr_binary_op(ctx: &mut Ctx, item: BinaryOp) -> Result<RegisterIndex, CompileError> {
    let op = item.op().ok_or_else(|| ice("binary op without operator"))?;

    // Short-circuit logical operators
    match op {
        BinaryOperator::And => return compile_logical_and(ctx, item),
        BinaryOperator::Or => return compile_logical_or(ctx, item),
        _ => {}
    }

    // Property access: a.b
    if op == BinaryOperator::Property {
        let lhs = item.lhs().ok_or_else(|| ice("binop without lhs"))?;
        let rhs = item.rhs().ok_or_else(|| ice("binop without rhs"))?;
        let table = compile_expr(ctx, lhs)?;
        let key = compile_property_key(ctx, rhs)?;
        let dst = ctx.alloc_register()?;
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
    let lhs = compile_expr(ctx, lhs_expr)?;
    let rhs = compile_expr(ctx, rhs_expr)?;

    // Arithmetic and bitwise operations
    match op {
        BinaryOperator::Add => emit_arith(ctx, lhs, rhs, Instruction::ADD { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::ADD),
        BinaryOperator::Sub => emit_arith(ctx, lhs, rhs, Instruction::SUB { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::SUB),
        BinaryOperator::Mul => emit_arith(ctx, lhs, rhs, Instruction::MUL { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::MUL),
        BinaryOperator::Div => emit_arith(ctx, lhs, rhs, Instruction::DIV { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::DIV),
        BinaryOperator::IntDiv => emit_arith(ctx, lhs, rhs, Instruction::IDIV { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::IDIV),
        BinaryOperator::Mod => emit_arith(ctx, lhs, rhs, Instruction::MOD { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::MOD),
        BinaryOperator::Exp => emit_arith(ctx, lhs, rhs, Instruction::POW { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::POW),
        BinaryOperator::BitAnd => emit_arith(ctx, lhs, rhs, Instruction::BAND { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::BAND),
        BinaryOperator::BitOr => emit_arith(ctx, lhs, rhs, Instruction::BOR { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::BOR),
        BinaryOperator::BitXor => emit_arith(ctx, lhs, rhs, Instruction::BXOR { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::BXOR),
        BinaryOperator::LShift => emit_arith(ctx, lhs, rhs, Instruction::SHL { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::SHL),
        BinaryOperator::RShift => emit_arith(ctx, lhs, rhs, Instruction::SHR { dst: 0, lhs: lhs.0, rhs: rhs.0 }, MetaMethod::SHR),
        BinaryOperator::Concat => {
            let dst = ctx.alloc_register()?;
            ctx.emit(Instruction::CONCAT { dst: dst.0, lhs: lhs.0, rhs: rhs.0 });
            Ok(dst)
        }

        // Comparisons — these produce a boolean result
        BinaryOperator::Eq => compile_comparison(ctx, lhs, rhs, |l, r| Instruction::EQ { lhs: l, rhs: r, inverted: false }),
        BinaryOperator::NEq => compile_comparison(ctx, lhs, rhs, |l, r| Instruction::EQ { lhs: l, rhs: r, inverted: true }),
        BinaryOperator::Lt => compile_comparison(ctx, lhs, rhs, |l, r| Instruction::LT { lhs: l, rhs: r, inverted: false }),
        BinaryOperator::Gt => compile_comparison(ctx, lhs, rhs, |l, r| Instruction::LT { lhs: r, rhs: l, inverted: false }),
        BinaryOperator::LEq => compile_comparison(ctx, lhs, rhs, |l, r| Instruction::LE { lhs: l, rhs: r, inverted: false }),
        BinaryOperator::GEq => compile_comparison(ctx, lhs, rhs, |l, r| Instruction::LE { lhs: r, rhs: l, inverted: false }),

        BinaryOperator::And | BinaryOperator::Or => unreachable!("handled above"),
        BinaryOperator::Property | BinaryOperator::Method => unreachable!("handled above"),
    }
}

fn emit_arith(
    ctx: &mut Ctx,
    lhs: RegisterIndex,
    rhs: RegisterIndex,
    mut instr: Instruction,
    metamethod: MetaMethod,
) -> Result<RegisterIndex, CompileError> {
    let dst = ctx.alloc_register()?;
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
    ctx.emit(Instruction::MMBIN {
        lhs: lhs.0,
        rhs: rhs.0,
        metamethod,
    });
    Ok(dst)
}

fn compile_comparison(
    ctx: &mut Ctx,
    lhs: RegisterIndex,
    rhs: RegisterIndex,
    make_cmp: impl FnOnce(u8, u8) -> Instruction,
) -> Result<RegisterIndex, CompileError> {
    let dst = ctx.alloc_register()?;

    // Emit: load true, comparison (skip next if true), LFALSESKIP, JMP over
    let true_idx = ctx.alloc_constant(Value::Boolean(true))?;
    ctx.emit(Instruction::LOAD {
        dst: dst.0,
        idx: true_idx,
    });
    ctx.emit(make_cmp(lhs.0, rhs.0));
    // If comparison is true, skip the next instruction (which loads false)
    // If false, execute the LFALSESKIP which sets to false and skips the jump
    ctx.emit(Instruction::LFALSESKIP { src: dst.0 });

    Ok(dst)
}

fn compile_logical_and(ctx: &mut Ctx, item: BinaryOp) -> Result<RegisterIndex, CompileError> {
    let lhs_expr = item.lhs().ok_or_else(|| ice("and without lhs"))?;
    let lhs = compile_expr(ctx, lhs_expr)?;

    let dst = ctx.alloc_register()?;
    ctx.emit(Instruction::MOVE {
        dst: dst.0,
        src: lhs.0,
    });

    // TEST: if lhs is falsy, skip JMP (short-circuit, result is lhs)
    let end_label = ctx.new_label();
    ctx.emit(Instruction::TEST {
        src: dst.0,
        inverted: true,
    });
    ctx.emit_jump(end_label);

    let rhs_expr = item.rhs().ok_or_else(|| ice("and without rhs"))?;
    let rhs = compile_expr(ctx, rhs_expr)?;
    ctx.emit(Instruction::MOVE {
        dst: dst.0,
        src: rhs.0,
    });

    ctx.set_label(end_label, ctx.next_offset());
    Ok(dst)
}

fn compile_logical_or(ctx: &mut Ctx, item: BinaryOp) -> Result<RegisterIndex, CompileError> {
    let lhs_expr = item.lhs().ok_or_else(|| ice("or without lhs"))?;
    let lhs = compile_expr(ctx, lhs_expr)?;

    let dst = ctx.alloc_register()?;
    ctx.emit(Instruction::MOVE {
        dst: dst.0,
        src: lhs.0,
    });

    // TEST: if lhs is truthy, skip JMP (short-circuit, result is lhs)
    let end_label = ctx.new_label();
    ctx.emit(Instruction::TEST {
        src: dst.0,
        inverted: false,
    });
    ctx.emit_jump(end_label);

    let rhs_expr = item.rhs().ok_or_else(|| ice("or without rhs"))?;
    let rhs = compile_expr(ctx, rhs_expr)?;
    ctx.emit(Instruction::MOVE {
        dst: dst.0,
        src: rhs.0,
    });

    ctx.set_label(end_label, ctx.next_offset());
    Ok(dst)
}

fn compile_expr_func_call(
    ctx: &mut Ctx,
    item: FuncCall,
    want: usize,
) -> Result<Vec<RegisterIndex>, CompileError> {
    let target = item
        .target()
        .ok_or_else(|| ice("func call without target"))?;
    let func = compile_expr(ctx, target)?;

    let args: Vec<_> = item.args().map(|a| a.collect()).unwrap_or_default();
    let nargs = args.len();

    // Compile arguments into consecutive registers after func
    for (i, arg_expr) in args.into_iter().enumerate() {
        let arg = compile_expr(ctx, arg_expr)?;
        let expected_reg = RegisterIndex(func.0 + 1 + i as u8);
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

    let object = compile_expr(ctx, object_expr)?;

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
        let arg = compile_expr(ctx, arg_expr)?;
        let expected_reg = RegisterIndex(func.0 + 2 + i as u8);
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

fn compile_expr_index(ctx: &mut Ctx, item: Index) -> Result<RegisterIndex, CompileError> {
    let target_expr = item
        .target()
        .ok_or_else(|| ice("index without target"))?;
    let key_expr = item.index().ok_or_else(|| ice("index without key"))?;
    let table = compile_expr(ctx, target_expr)?;
    let key = compile_expr(ctx, key_expr)?;
    let dst = ctx.alloc_register()?;
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
        let reg = compile_expr(ctx, expr)?;
        let target = RegisterIndex(first_reg.0 + i as u8);
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

fn compile_while(ctx: &mut Ctx, item: While) -> Result<(), CompileError> {
    scope_lexical_break(ctx, |ctx| {
        let loop_start = ctx.new_label();
        ctx.set_label(loop_start, ctx.next_offset());

        let cond_expr = item
            .cond()
            .ok_or_else(|| ice("while without condition"))?;
        let cond = compile_expr(ctx, cond_expr)?;

        // If condition is falsy, jump to break label (end of loop)
        let break_label = *ctx
            .control_end_label
            .last()
            .ok_or_else(|| ice("missing break label"))?;
        ctx.emit(Instruction::TEST {
            src: cond.0,
            inverted: true,
        });
        ctx.emit_jump(break_label);

        // Compile body
        if let Some(block) = item.block() {
            let stmts: Vec<_> = block.stmts().map(|s| s.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }
        }

        // Jump back to condition check
        ctx.emit_jump(loop_start);
        Ok(())
    })
}

fn compile_repeat(ctx: &mut Ctx, item: Repeat) -> Result<(), CompileError> {
    scope_break(ctx, |ctx| {
        // Note: repeat-until has special scoping — the condition can see
        // locals defined in the body. So we wrap the body AND condition
        // in the same lexical scope.
        scope_lexical(ctx, |ctx| {
            let loop_start = ctx.new_label();
            ctx.set_label(loop_start, ctx.next_offset());

            // Compile body
            let stmts: Vec<_> = item.block().map(|b| b.collect()).unwrap_or_default();
            for stmt in stmts {
                compile_stmt(ctx, stmt)?;
            }

            // Compile condition
            let cond_expr = item
                .cond()
                .ok_or_else(|| ice("repeat without condition"))?;
            let cond = compile_expr(ctx, cond_expr)?;

            // If condition is falsy, loop back
            ctx.emit(Instruction::TEST {
                src: cond.0,
                inverted: true,
            });
            ctx.emit_jump(loop_start);

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
    let cond = compile_expr(ctx, cond_expr)?;

    let else_label = ctx.new_label();
    ctx.emit(Instruction::TEST {
        src: cond.0,
        inverted: true,
    });
    ctx.emit_jump(else_label);

    // Then block
    scope_lexical(ctx, |ctx| {
        let stmts: Vec<_> = item.stmts().map(|s| s.collect()).unwrap_or_default();
        for stmt in stmts {
            compile_stmt(ctx, stmt)?;
        }
        Ok(())
    })?;

    ctx.emit_jump(end_label);
    ctx.set_label(else_label, ctx.next_offset());

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

        let limit_expr = item
            .end()
            .ok_or_else(|| ice("for_num without limit"))?;

        // Compile init, limit, step into consecutive registers: base, base+1, base+2
        let init = compile_expr(ctx, init_expr)?;
        let base = ctx.alloc_register()?;
        if init != base {
            ctx.emit(Instruction::MOVE {
                dst: base.0,
                src: init.0,
            });
        }

        let limit = compile_expr(ctx, limit_expr)?;
        let limit_reg = ctx.alloc_register()?;
        if limit != limit_reg {
            ctx.emit(Instruction::MOVE {
                dst: limit_reg.0,
                src: limit.0,
            });
        }

        let step_reg = ctx.alloc_register()?;
        if let Some(step_expr) = item.step() {
            let step = compile_expr(ctx, step_expr)?;
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

        // Compile up to 3 iterator values
        for (i, val_expr) in values.into_iter().enumerate().take(3) {
            let val = compile_expr(ctx, val_expr)?;
            let target_reg = RegisterIndex(base.0 + i as u8);
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

        // Ensure we have at least 3 control registers
        while ctx.chunk.register_count < base.0 as usize + 3 {
            ctx.alloc_register()?;
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
