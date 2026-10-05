//! `string.format`'s float conversions (the LuaJIT `lj_strfmt_num` port) and `%.14g`
//! `tostring`. Expected outputs from lua 5.5.1, except where noted.

use crate::common::{err, ok};

#[test]
fn exact_ties_round_to_even() {
    assert_eq!(
        ok(
            "return string.format('%.0e %.1e %.0e %.12e %.2f %.0f %.0f', \
            2.5e10, 2.25e12, 25, 2^-20, 0.125, 0.5, 1.5)"
        ),
        "2e+10 2.2e+12 2e+01 9.536743164062e-07 0.12 0 2"
    );
    // Not a tie: the digits past the 5 are nonzero, though the rescaled conversion's lowest limb
    // isn't.
    assert_eq!(
        ok("return string.format('%.13e|%.14g', 0x1.e3c5d95a33ebp-977, 0x1.e3c5d95a33ebp-977)"),
        "1.4794345626215e-294|1.4794345626215e-294"
    );
}

#[test]
fn extremes() {
    assert_eq!(
        ok("return string.format('%f %.3f %f %e %g', 2^-1022, -2^-1074, 1e-300, 2^-1074, 2^-1074)"),
        "0.000000 -0.000 0.000000 4.940656e-324 4.94066e-324"
    );
    assert_eq!(
        ok("return string.format('%.99f', 0.1)"),
        "0.100000000000000005551115123125782702118158340454101562500000000000000000000000000000000000000000000"
    );
    assert_eq!(
        ok("return string.format('%.0f', 2^1023)"),
        "89884656743115795386465259539451236680898848947115328636715040578866337902750481566354238661203768010560056939935696678829394884407208311246423715319737062188883946712432742638151109800623047059726541476042502884419075341171231440736956555270413618581675255342293149119973622969239858152417678164812112068608"
    );
    assert_eq!(
        ok("return string.format('%f %e %g %5.1f %-6f|', 1/0, -1/0, 0/0, 1/0, -1/0)"),
        "inf -inf nan   inf -inf  |"
    );
}

#[test]
fn flags_width_precision() {
    assert_eq!(
        ok(
            "return string.format('%5.40f|%-12g|%012.3e|%+#.0f|% g|%#g|%G', \
            1/3, 1e-5, -123.456, 2.0, 1e20, 1.5, 1e-20)"
        ),
        "0.3333333333333333148296162562473909929395|1e-05       |-001.235e+02|+2.| 1e+20|1.50000|1E-20"
    );
    assert_eq!(
        ok("return string.format('%a %A %.3a %a %a', 1.0, 0.5, 1.0, 2^-1074, 0.0)"),
        "0x1p+0 0X1P-1 0x1.000p+0 0x1p-1074 0x0p+0"
    );
}

#[test]
fn no_uppercase_f() {
    assert_eq!(
        err("string.format('%F', 1)"),
        "c:1: invalid conversion '%F' to 'format'"
    );
    assert_eq!(
        err("string.format('%-5.2F', 1)"),
        "c:1: invalid conversion '%-5.2F' to 'format'"
    );
}

#[test]
fn tostring_is_14_digits() {
    // Intentional divergence: LuaJIT's `%.14g`, where Lua 5.5 prints `0.30000000000000004`,
    // `0.33333333333333331` and `9.2233720368547758e+18`.
    assert_eq!(
        ok("return cat(0.1 + 0.2, 1/3, 2^63, -0.0, 100.0, 1e15, 1e100, 2^53)"),
        "0.3 0.33333333333333 9.2233720368548e+18 -0.0 100.0 1e+15 1e+100 9.007199254741e+15"
    );
}
