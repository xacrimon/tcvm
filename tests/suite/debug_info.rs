//! Prototype debug info: per-instruction lines, function line spans, local
//! variable ranges, upvalue names, and the chunk name from `load`.
//! Expected values were taken from `luac -l -l` on the same source.

use tcvm::Lua;
use tcvm::env::Prototype;

const SRC: &str = "local a = 1\n\
                   local b = 2\n\
                   do\n\
                     local c = a + b\n\
                     print(c)\n\
                   end\n\
                   local function f(x)\n\
                     return x\n\
                   end\n";

fn with_proto(f: impl for<'gc> FnOnce(&Prototype<'gc>)) {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx.load(SRC, Some("=locs")).expect("load");
        let closure = chunk.as_lua().expect("lua closure");
        f(&closure.proto);
    });
}

#[test]
fn lines_per_instruction() {
    with_proto(|proto| {
        let lines: Vec<u32> = (0..proto.code.len())
            .map(|pc| proto.line_for_pc(pc).unwrap())
            .collect();
        // VARARGPREP LOAD LOAD ADD GETTABUP MOVE CALL CLOSURE RETURN
        assert_eq!(lines, [1, 1, 2, 4, 5, 5, 5, 9, 9]);
        assert_eq!(proto.line_for_pc(proto.code.len()), None);
    });
}

#[test]
fn function_line_span_and_source() {
    with_proto(|proto| {
        assert_eq!((proto.line_defined, proto.last_line_defined), (0, 0));
        assert_eq!(proto.source.as_bytes(), b"=locs");
        let f = &proto.prototypes[0];
        assert_eq!((f.line_defined, f.last_line_defined), (7, 9));
        assert_eq!(f.source.as_bytes(), b"=locs");
        assert_eq!(f.line_for_pc(0), Some(8));
    });
}

#[test]
fn local_variable_ranges() {
    with_proto(|proto| {
        let vars: Vec<(&[u8], u32, u32)> = proto
            .locvars
            .iter()
            .map(|v| (v.name.as_bytes(), v.start_pc, v.end_pc))
            .collect();
        let end = proto.code.len() as u32;
        assert_eq!(
            &vars[..3],
            [(&b"a"[..], 2, end), (b"b", 3, end), (b"c", 4, 7)]
        );
        // `local function f` is visible to debug info only after CLOSURE.
        assert_eq!((vars[3].0, vars[3].1, vars[3].2), (&b"f"[..], 8, end));
        let f = &proto.prototypes[0];
        assert_eq!(f.locvars.len(), 1);
        assert_eq!(f.locvars[0].name.as_bytes(), b"x");
        assert_eq!(f.locvars[0].start_pc, 0);
    });
}

#[test]
fn upvalue_names() {
    with_proto(|proto| {
        let names: Vec<&[u8]> = proto.upvalue_names.iter().map(|n| n.as_bytes()).collect();
        assert_eq!(names, [b"_ENV"]);
        assert_eq!(proto.upvalue_names.len(), proto.upvalue_desc.len());
    });
}

#[test]
fn global_declarations_are_not_locals() {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx
            .load(
                "global x\nx = 1\nlocal y = 2\nglobal function g() end",
                Some("=g"),
            )
            .expect("load");
        let proto = chunk.as_lua().unwrap().proto;
        let names: Vec<&[u8]> = proto.locvars.iter().map(|v| v.name.as_bytes()).collect();
        assert_eq!(names, [b"y"]);
    });
}

#[test]
fn loop_control_slots_are_recorded() {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx
            .load(
                "for i = 1, 2 do end\nfor k, v in next, {} do end",
                Some("=l"),
            )
            .expect("load");
        let proto = chunk.as_lua().unwrap().proto;
        let names: Vec<&[u8]> = proto.locvars.iter().map(|v| v.name.as_bytes()).collect();
        let fs = &b"(for state)"[..];
        assert_eq!(names, [fs, fs, fs, b"i", fs, fs, fs, b"k", b"v"]);
        // Control slots live from FORPREP (pc 4) through FORLOOP (pc 5);
        // the visible variable only inside the (empty) body.
        assert_eq!((proto.locvars[0].start_pc, proto.locvars[0].end_pc), (4, 6));
        assert_eq!((proto.locvars[3].start_pc, proto.locvars[3].end_pc), (5, 5));
        // Generic loop: TFORPREP at 10, TFORLOOP at 12.
        assert_eq!(
            (proto.locvars[4].start_pc, proto.locvars[4].end_pc),
            (10, 13)
        );
    });
}

#[test]
fn unnamed_chunks_are_named_by_their_source() {
    let mut lua = Lua::new();
    lua.load_all();
    lua.enter(|ctx| {
        let chunk = ctx.load("return 1", None).expect("load");
        assert_eq!(chunk.as_lua().unwrap().proto.source.as_bytes(), b"return 1");
    });
}
