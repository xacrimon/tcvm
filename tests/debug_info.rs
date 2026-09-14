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
        assert_eq!(proto.source.unwrap().as_bytes(), b"=locs");
        let f = &proto.prototypes[0];
        assert_eq!((f.line_defined, f.last_line_defined), (7, 9));
        assert_eq!(f.source.unwrap().as_bytes(), b"=locs");
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
        assert_eq!(vars[3].0, b"f");
        assert_eq!(vars[3].2, end);
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
