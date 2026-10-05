//! `%a`/`%e`/`%f`/`%g` for floats, ported from LuaJIT's `lj_strfmt_num.c`
//! (<https://github.com/LuaJIT/LuaJIT>, commit c6ffc141a8762b41703f9287d63d93622a13dd8f, MIT,
//! Copyright (C) 2005-2026 Mike Pall, contributed by Peter Cawley), with `lj_strfmt_wint` from
//! `lj_strfmt.c`. Diff against that commit when porting upstream fixes.
//!
//! Deviations: output goes to a `Vec<u8>` grown by the same amounts `lj_buf_more` reserves, with
//! an index for the C pointer; only the 64-bit `ND_MUL2K_MAX_SHIFT` of 29 is kept; `nd_round`
//! learns whether a 5 is an exact tie from `n` itself, where upstream scans limbs that the
//! division stops, the multiplication shortcut and the rescale check leave inexact (it rounds
//! `%.0e` of 2.5e10 up and, depending on stack garbage, `%.13e` of 0x1.e3c5d95a33ebp-977 down);
//! and `%f` of a number below 2^-500 is zero outright, where upstream's unbounded division
//! wraps the 64-limb ring into the integer part (`%f` of 2^-1022 prints `906459071.863478`).

/// `SFormat`: conversion, flags, width and precision packed as in `lj_strfmt.h`.
pub(crate) type SFormat = u32;

pub(crate) const STRFMT_T_FP_A: SFormat = 0x0000;
pub(crate) const STRFMT_T_FP_E: SFormat = 0x0010;
pub(crate) const STRFMT_T_FP_F: SFormat = 0x0020;
pub(crate) const STRFMT_T_FP_G: SFormat = 0x0030;

pub(crate) const STRFMT_F_LEFT: SFormat = 0x0100;
pub(crate) const STRFMT_F_PLUS: SFormat = 0x0200;
pub(crate) const STRFMT_F_ZERO: SFormat = 0x0400;
pub(crate) const STRFMT_F_SPACE: SFormat = 0x0800;
pub(crate) const STRFMT_F_ALT: SFormat = 0x1000;
pub(crate) const STRFMT_F_UPPER: SFormat = 0x2000;

pub(crate) const STRFMT_SH_WIDTH: u32 = 16;
pub(crate) const STRFMT_SH_PREC: u32 = 24;

/// LuaJIT's `tostring` format, `%.14g`.
pub(crate) const STRFMT_G14: SFormat = STRFMT_T_FP_G | ((14 + 1) << STRFMT_SH_PREC);

#[inline]
fn strfmt_width(sf: SFormat) -> u32 {
    (sf >> STRFMT_SH_WIDTH) & 255
}

/// `!0` when unset.
#[inline]
fn strfmt_prec(sf: SFormat) -> u32 {
    ((sf >> STRFMT_SH_PREC) & 255).wrapping_sub(1)
}

#[inline]
fn strfmt_fp(sf: SFormat) -> u32 {
    (sf >> 4) & 3
}

#[inline]
fn lj_ffs(x: u32) -> u32 {
    x.trailing_zeros()
}

#[inline]
fn lj_fls(x: u32) -> u32 {
    x.leading_zeros() ^ 31
}

#[inline]
fn hi(t: u64) -> u32 {
    (t >> 32) as u32
}

#[inline]
fn lo(t: u64) -> u32 {
    t as u32
}

// -- Precomputed tables --------------------------------------------------

/// Rescale factors to push the exponent of a number towards zero.
static RESCALE_E: [i16; 32] = [
    -308, -289, -270, -250, -231, -212, -193, -173, -154, -135, -115, -96, -77, -58, -38, 0, 0, 0,
    39, 58, 77, 96, 116, 135, 154, 174, 193, 212, 231, 251, 270, 289,
];
static RESCALE_N: [f64; 32] = [
    1e308, 1e289, 1e270, 1e250, 1e231, 1e212, 1e193, 1e173, 1e154, 1e135, 1e115, 1e96, 1e77, 1e58,
    1e38, 1e0, 1e0, 1e0, 1e-39, 1e-58, 1e-77, 1e-96, 1e-116, 1e-135, 1e-154, 1e-174, 1e-193,
    1e-212, 1e-231, 1e-251, 1e-270, 1e-289,
];

/// For p in range -70 through 57, this table encodes pairs (m, e) such that
/// 4*2^p <= (uint8_t)m*10^e, and is the smallest value for which this holds.
static FOUR_ULP_M_E: [i8; 256] = [
    34, -21, 68, -21, 14, -20, 28, -20, 55, -20, 2, -19, 3, -19, 5, -19, 9, -19, -82, -18, 35, -18,
    7, -17, -117, -17, 28, -17, 56, -17, 112, -16, -33, -16, 45, -16, 89, -16, -78, -15, 36, -15,
    72, -15, -113, -14, 29, -14, 57, -14, 114, -13, -28, -13, 46, -13, 91, -12, -74, -12, 37, -12,
    73, -12, 15, -11, 3, -11, 59, -11, 2, -10, 3, -10, 5, -10, 1, -9, -69, -9, 38, -9, 75, -9, 15,
    -7, 3, -7, 6, -7, 12, -6, -17, -7, 48, -7, 96, -7, -65, -6, 39, -6, 77, -6, -103, -5, 31, -5,
    62, -5, 123, -4, -11, -4, 49, -4, 98, -4, -60, -3, 4, -2, 79, -3, 16, -2, 32, -2, 63, -2, 2,
    -1, 25, 0, 5, 1, 1, 2, 2, 2, 4, 2, 8, 2, 16, 2, 32, 2, 64, 2, -128, 2, 26, 2, 52, 2, 103, 3,
    -51, 3, 41, 4, 82, 4, -92, 4, 33, 4, 66, 4, -124, 5, 27, 5, 53, 5, 105, 6, 21, 6, 42, 6, 84, 6,
    17, 7, 34, 7, 68, 7, 2, 8, 3, 8, 6, 8, 108, 9, -41, 9, 43, 10, 86, 9, -84, 10, 35, 10, 69, 10,
    -118, 11, 28, 11, 55, 12, 11, 13, 22, 13, 44, 13, 88, 13, -80, 13, 36, 13, 71, 13, -115, 14,
    29, 14, 57, 14, 113, 15, -30, 15, 46, 15, 91, 15, 19, 16, 37, 16, 73, 16, 2, 17, 3, 17, 6, 17,
];

/// min(2^32-1, 10^e-1) for e in range 0 through 10
static NDIGITS_DEC_THRESHOLD: [u32; 11] = [
    0, 9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, 999999999, 0xffffffff,
];

// -- Helper functions ----------------------------------------------------

/// Compute the number of digits in the decimal representation of x.
#[inline]
fn ndigits_dec(x: u32) -> u32 {
    let t = ((lj_fls(x | 1) * 77) >> 8) + 1; // 2^8/77 is roughly log2(10)
    t + (x > NDIGITS_DEC_THRESHOLD[t as usize]) as u32
}

macro_rules! wint_r {
    ($b:ident, $p:ident, $x:ident, $sh:expr, $sc:expr) => {{
        let d = ($x * (1u32 << $sh).div_ceil($sc)) >> $sh;
        $x -= d * $sc;
        $b[$p] = b'0' + d as u8;
        $p += 1;
    }};
}

/// Write 9-digit unsigned integer to buffer.
fn wuint9(b: &mut [u8], mut p: usize, mut u: u32) -> usize {
    let mut v = u / 10000;
    u -= v * 10000;
    let w = v / 10000;
    v -= w * 10000;
    b[p] = b'0' + w as u8;
    p += 1;
    wint_r!(b, p, v, 23, 1000);
    wint_r!(b, p, v, 12, 100);
    wint_r!(b, p, v, 10, 10);
    b[p] = b'0' + v as u8;
    p += 1;
    wint_r!(b, p, u, 23, 1000);
    wint_r!(b, p, u, 12, 100);
    wint_r!(b, p, u, 10, 10);
    b[p] = b'0' + u as u8;
    p + 1
}

/// Write integer to buffer (`lj_strfmt_wint`; its `goto`s become digit counts).
fn wint(b: &mut [u8], mut p: usize, k: i32) -> usize {
    let mut u = k as u32;
    if k < 0 {
        u = (!u).wrapping_add(1);
        b[p] = b'-';
        p += 1;
    }
    let udigits = if u < 10000 {
        if u < 10 {
            1
        } else if u < 100 {
            2
        } else if u < 1000 {
            3
        } else {
            4
        }
    } else {
        let mut v = u / 10000;
        u -= v * 10000;
        let vdigits = if v < 10000 {
            if v < 10 {
                1
            } else if v < 100 {
                2
            } else if v < 1000 {
                3
            } else {
                4
            }
        } else {
            let mut w = v / 10000;
            v -= w * 10000;
            if w >= 10 {
                wint_r!(b, p, w, 10, 10);
            }
            b[p] = b'0' + w as u8;
            p += 1;
            4
        };
        if vdigits >= 4 {
            wint_r!(b, p, v, 23, 1000);
        }
        if vdigits >= 3 {
            wint_r!(b, p, v, 12, 100);
        }
        if vdigits >= 2 {
            wint_r!(b, p, v, 10, 10);
        }
        b[p] = b'0' + v as u8;
        p += 1;
        4
    };
    if udigits >= 4 {
        wint_r!(b, p, u, 23, 1000);
    }
    if udigits >= 3 {
        wint_r!(b, p, u, 12, 100);
    }
    if udigits >= 2 {
        wint_r!(b, p, u, 10, 10);
    }
    b[p] = b'0' + u as u8;
    p + 1
}

// -- Extended precision arithmetic ---------------------------------------

// The "nd" format is a fixed-precision decimal representation for numbers. It
// consists of up to 64 uint32_t values, with each uint32_t storing a value
// in the range [0, 1e9). A number in "nd" format consists of three variables:
//
//  uint32_t nd[64];
//  uint32_t ndlo;
//  uint32_t ndhi;
//
// The integral part of the number is stored in nd[0 ... ndhi], the value of
// which is sum{i in [0, ndhi] | nd[i] * 10^(9*i)}. If the fractional part of
// the number is zero, ndlo is zero. Otherwise, the fractional part is stored
// in nd[ndlo ... 63], the value of which is taken to be
// sum{i in [ndlo, 63] | nd[i] * 10^(9*(i-64))}.
//
// If the array part had 128 elements rather than 64, then every double would
// have an exact representation in "nd" format. With 64 elements, all integral
// doubles have an exact representation, and all non-integral doubles have
// enough digits to make both %.99e and %.99f do the right thing.

const ND_MUL2K_MAX_SHIFT: u32 = 29;

/// Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero).
fn nd_mul2k(nd: &mut [u32; 64], mut ndhi: u32, mut k: u32, mut carry_in: u32, sf: SFormat) -> u32 {
    let mut ndlo = 0u32;
    let mut start = 1u32;
    // Performance hacks.
    if k > ND_MUL2K_MAX_SHIFT * 2 && strfmt_fp(sf) != strfmt_fp(STRFMT_T_FP_F) {
        start = ndhi.wrapping_sub(strfmt_prec(sf).wrapping_add(17) / 8);
    }
    // Real logic.
    while k >= ND_MUL2K_MAX_SHIFT {
        for i in ndlo..=ndhi {
            let val = ((nd[i as usize] as u64) << ND_MUL2K_MAX_SHIFT) | carry_in as u64;
            carry_in = (val / 1000000000) as u32;
            nd[i as usize] = (val as u32).wrapping_sub(carry_in.wrapping_mul(1000000000));
        }
        if carry_in != 0 {
            ndhi += 1;
            nd[ndhi as usize] = carry_in;
            carry_in = 0;
            if start == ndlo {
                ndlo += 1;
            }
            start = start.wrapping_add(1);
        }
        k -= ND_MUL2K_MAX_SHIFT;
    }
    if k != 0 {
        for i in ndlo..=ndhi {
            let val = ((nd[i as usize] as u64) << k) | carry_in as u64;
            carry_in = (val / 1000000000) as u32;
            nd[i as usize] = (val as u32).wrapping_sub(carry_in.wrapping_mul(1000000000));
        }
        if carry_in != 0 {
            ndhi += 1;
            nd[ndhi as usize] = carry_in;
        }
    }
    ndhi
}

/// Divide nd by 2^k (ndlo is assumed to be zero).
fn nd_div2k(nd: &mut [u32; 64], mut ndhi: u32, mut k: u32, sf: SFormat) -> u32 {
    let mut ndlo = 0u32;
    let mut stop1 = !0u32;
    let mut stop2 = !0u32;
    // Performance hacks.
    if ndhi == 0 {
        if nd[0] == 0 {
            return 0;
        } else {
            let s = lj_ffs(nd[0]);
            if s >= k {
                nd[0] >>= k;
                return 0;
            }
            nd[0] >>= s;
            k -= s;
        }
    }
    if k > 18 {
        if strfmt_fp(sf) == strfmt_fp(STRFMT_T_FP_F) {
            // Must not limit precision here or nd_round cannot round to even.
            // stop1 = 63 - (int32_t)STRFMT_PREC(sf) / 9;
        } else {
            let floorlog2 = (ndhi * 29 + lj_fls(nd[ndhi as usize])).wrapping_sub(k) as i32;
            let floorlog10 = (floorlog2 as f64 * 0.30102999566398114) as i32;
            stop1 = (62 + (floorlog10.wrapping_sub(strfmt_prec(sf) as i32)) / 9) as u32;
            stop2 = (61 + ndhi).wrapping_sub(((strfmt_prec(sf) as i32) / 8) as u32);
        }
    }
    // Real logic.
    while k >= 9 {
        let mut i = ndhi;
        let mut carry = 0u32;
        loop {
            let val = nd[i as usize];
            nd[i as usize] = (val >> 9) + carry;
            carry = (val & 0x1ff) * 1953125;
            if i == ndlo {
                break;
            }
            i = i.wrapping_sub(1) & 0x3f;
        }
        if ndlo != stop1 && ndlo != stop2 {
            if carry != 0 {
                ndlo = ndlo.wrapping_sub(1) & 0x3f;
                nd[ndlo as usize] = carry;
            }
            if nd[ndhi as usize] == 0 {
                ndhi = ndhi.wrapping_sub(1) & 0x3f;
                stop2 = stop2.wrapping_sub(1);
            }
        } else if nd[ndhi as usize] == 0 {
            if ndhi != ndlo {
                ndhi = ndhi.wrapping_sub(1) & 0x3f;
                stop2 = stop2.wrapping_sub(1);
            } else {
                return ndlo;
            }
        }
        k -= 9;
    }
    if k != 0 {
        let mask = (1u32 << k) - 1;
        let mul = 1000000000 >> k;
        let mut i = ndhi;
        let mut carry = 0u32;
        loop {
            let val = nd[i as usize];
            nd[i as usize] = (val >> k) + carry;
            carry = (val & mask) * mul;
            if i == ndlo {
                break;
            }
            i = i.wrapping_sub(1) & 0x3f;
        }
        if carry != 0 {
            ndlo = ndlo.wrapping_sub(1) & 0x3f;
            nd[ndlo as usize] = carry;
        }
    }
    ndlo
}

/// Add m*10^e to nd (assumes ndlo <= e/9 <= ndhi and 0 <= m <= 9).
fn nd_add_m10e(nd: &mut [u32; 64], mut ndhi: u32, m: u8, e: i32) -> u32 {
    let mut i: u32;
    let mut carry: u32;
    if e >= 0 {
        i = e as u32 / 9;
        carry = (m as u32).wrapping_mul(NDIGITS_DEC_THRESHOLD[(e - i as i32 * 9) as usize] + 1);
    } else {
        let f = (e - 8) / 9;
        i = (64 + f) as u32;
        carry = (m as u32).wrapping_mul(NDIGITS_DEC_THRESHOLD[(e - f * 9) as usize] + 1);
    }
    loop {
        let mut val = nd[i as usize].wrapping_add(carry);
        if val >= 1000000000 {
            val -= 1000000000;
            nd[i as usize] = val;
            if i == ndhi {
                ndhi = (ndhi + 1) & 0x3f;
                nd[ndhi as usize] = 1;
                break;
            }
            carry = 1;
            i = (i + 1) & 0x3f;
        } else {
            nd[i as usize] = val;
            break;
        }
    }
    ndhi
}

/// Round to even with given precision. Extra digits are not zeroed. `tail_zero` says whether
/// every digit below 10^e is zero, which `nd` itself can't tell (see [`multiple_of_pow10`]).
fn nd_round(nd: &mut [u32; 64], ndhi: u32, e: i32, tail_zero: bool) -> u32 {
    let i: u32;
    let d: i32;
    let mut buf = [0u8; 9];
    if e >= 0 {
        i = e as u32 / 9;
        d = 8 - e + i as i32 * 9;
    } else {
        let f = (e - 8) / 9;
        i = (64 + f) as u32;
        d = 8 - e + f * 9;
    }
    wuint9(&mut buf, 0, nd[i as usize]);
    if buf[d as usize] < b'5' {
        return ndhi; // Don't round up.
    } else if buf[d as usize] == b'5' {
        // Must check for round to even.
        let odd = if d != 0 {
            buf[d as usize - 1] & 1 != 0
        } else {
            nd[((i + 1) & 0x3f) as usize] & 1 != 0
        };
        // Round up '[13579]5.*' and '[02468]5[^0]*'.
        if !odd && tail_zero {
            return ndhi; // Don't round up.
        }
    } // else: round up.
    nd_add_m10e(nd, ndhi, 5, e) // Round up by adding 5*10^e.
}

/// Whether finite `n` is a multiple of 10^k, i.e. all its digits below 10^k are zero. With
/// n = m*2^e and m odd, that is k <= e for k <= 0, and e >= k with 5^k dividing m for k > 0.
fn multiple_of_pow10(n: f64, k: i32) -> bool {
    let bits = n.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    let mut m = bits & 0x000f_ffff_ffff_ffff;
    let mut e = if biased == 0 { -1074 } else { biased - 1075 };
    if biased != 0 {
        m |= 1 << 52;
    }
    if m == 0 {
        return true;
    }
    e += m.trailing_zeros() as i32;
    m >>= m.trailing_zeros();
    if k <= 0 {
        k <= e
    } else {
        // 5^k divides m < 2^53 only for k <= 22.
        e >= k && k <= 22 && m.is_multiple_of(5u64.pow(k as u32))
    }
}

/// Test whether two "nd" values are equal in their most significant digits.
fn nd_similar(nd: &[u32; 64], mut ndhi: u32, mut r: usize, hilen: u32, mut prec: u32) -> bool {
    let mut nd9 = [0u8; 9];
    let mut ref9 = [0u8; 9];
    if hilen <= prec {
        if nd[ndhi as usize] != nd[r] {
            return false;
        }
        prec -= hilen;
        r -= 1;
        ndhi = ndhi.wrapping_sub(1) & 0x3f;
        if prec >= 9 {
            if nd[ndhi as usize] != nd[r] {
                return false;
            }
            prec -= 9;
            r -= 1;
            ndhi = ndhi.wrapping_sub(1) & 0x3f;
        }
    } else {
        prec = prec.wrapping_sub(hilen.wrapping_sub(9));
    }
    debug_assert!(prec < 9, "bad precision {prec}");
    wuint9(&mut nd9, 0, nd[ndhi as usize]);
    wuint9(&mut ref9, 0, nd[r]);
    let prec = prec as usize;
    nd9[..prec] == ref9[..prec] && (nd9[prec] < b'5') == (ref9[prec] < b'5')
}

// -- Formatted conversions to buffer -------------------------------------

/// `lj_buf_more`: grow `sb` by `sz` and return where the new bytes start.
#[inline]
fn buf_more(sb: &mut Vec<u8>, sz: u32) -> usize {
    let p = sb.len();
    sb.resize(p + sz as usize, 0);
    p
}

/// Append `n` formatted per `sf` (`lj_strfmt_putfnum`).
pub(crate) fn put_fnum(sb: &mut Vec<u8>, sf: SFormat, n: f64) {
    let end = wfnum(sb, sf, n);
    sb.truncate(end);
}

/// Write formatted floating-point number to sb, returning the end of the output.
fn wfnum(sb: &mut Vec<u8>, sf: SFormat, n: f64) -> usize {
    let mut width = strfmt_width(sf);
    let mut prec = strfmt_prec(sf);
    let len: u32;
    let mut t = n.to_bits();
    let mut p: usize;
    if (hi(t) << 1) >= 0xffe00000 {
        // Handle non-finite values uniformly for %a, %e, %f, %g.
        let mut prefix = 0u8;
        let mut ch: u32 = if sf & STRFMT_F_UPPER != 0 {
            0x202020
        } else {
            0
        };
        if ((hi(t) & 0x000fffff) | lo(t)) != 0 {
            ch ^= ((b'n' as u32) << 16) | ((b'a' as u32) << 8) | b'n' as u32;
            if sf & STRFMT_F_SPACE != 0 {
                prefix = b' ';
            }
        } else {
            ch ^= ((b'i' as u32) << 16) | ((b'n' as u32) << 8) | b'f' as u32;
            if hi(t) & 0x80000000 != 0 {
                prefix = b'-';
            } else if sf & STRFMT_F_PLUS != 0 {
                prefix = b'+';
            } else if sf & STRFMT_F_SPACE != 0 {
                prefix = b' ';
            }
        }
        len = 3 + (prefix != 0) as u32;
        p = buf_more(sb, width.max(len));
        if sf & STRFMT_F_LEFT == 0 {
            while width > len {
                width -= 1;
                sb[p] = b' ';
                p += 1;
            }
        }
        if prefix != 0 {
            sb[p] = prefix;
            p += 1;
        }
        sb[p] = (ch >> 16) as u8;
        sb[p + 1] = (ch >> 8) as u8;
        sb[p + 2] = ch as u8;
        p += 3;
    } else if strfmt_fp(sf) == strfmt_fp(STRFMT_T_FP_A) {
        // %a
        let hexdig: &[u8; 18] = if sf & STRFMT_F_UPPER != 0 {
            b"0123456789ABCDEFPX"
        } else {
            b"0123456789abcdefpx"
        };
        let mut e = ((hi(t) >> 20) & 0x7ff) as i32;
        let mut prefix = 0u8;
        let mut eprefix = b'+';
        if hi(t) & 0x80000000 != 0 {
            prefix = b'-';
        } else if sf & STRFMT_F_PLUS != 0 {
            prefix = b'+';
        } else if sf & STRFMT_F_SPACE != 0 {
            prefix = b' ';
        }
        t &= 0x000f_ffff_ffff_ffff;
        if e != 0 {
            t |= 1 << 52;
            e -= 1023;
        } else if t != 0 {
            // Non-zero denormal - normalise it.
            let shift = if hi(t) != 0 {
                20 - lj_fls(hi(t))
            } else {
                52 - lj_fls(lo(t))
            };
            e = -1022 - shift as i32;
            t <<= shift;
        }
        // abs(n) == t * 2^(e - 52)
        // If n != 0, bit 52 of t is set, and is the highest set bit.
        if (prec as i32) < 0 {
            // Default precision: use smallest precision giving exact result.
            prec = if lo(t) != 0 {
                13 - lj_ffs(lo(t)) / 4
            } else {
                5 - lj_ffs(hi(t) | 0x100000) / 4
            };
        } else if prec < 13 {
            // Precision is sufficiently low as to maybe require rounding.
            t += 1u64 << (51 - prec * 4);
        }
        if e < 0 {
            eprefix = b'-';
            e = -e;
        }
        len = 5
            + ndigits_dec(e as u32)
            + prec
            + (prefix != 0) as u32
            + ((prec | (sf & STRFMT_F_ALT)) != 0) as u32;
        p = buf_more(sb, width.max(len));
        if sf & (STRFMT_F_LEFT | STRFMT_F_ZERO) == 0 {
            while width > len {
                width -= 1;
                sb[p] = b' ';
                p += 1;
            }
        }
        if prefix != 0 {
            sb[p] = prefix;
            p += 1;
        }
        sb[p] = b'0';
        sb[p + 1] = hexdig[17]; // x or X
        p += 2;
        if sf & (STRFMT_F_LEFT | STRFMT_F_ZERO) == STRFMT_F_ZERO {
            while width > len {
                width -= 1;
                sb[p] = b'0';
                p += 1;
            }
        }
        sb[p] = b'0' + (hi(t) >> 20) as u8; // Usually '1', sometimes '0' or '2'.
        p += 1;
        if (prec | (sf & STRFMT_F_ALT)) != 0 {
            // Emit fractional part.
            let q = p + 1 + prec as usize;
            sb[p] = b'.';
            if prec < 13 {
                t >>= 52 - prec * 4;
            } else {
                while prec > 13 {
                    sb[p + prec as usize] = b'0';
                    prec -= 1;
                }
            }
            while prec != 0 {
                sb[p + prec as usize] = hexdig[(t & 15) as usize];
                t >>= 4;
                prec -= 1;
            }
            p = q;
        }
        sb[p] = hexdig[16]; // p or P
        sb[p + 1] = eprefix; // + or -
        p = wint(sb, p + 2, e);
    } else {
        // %e or %f or %g - begin by converting n to "nd" format.
        let mut nd = [0u32; 64];
        let mut ndhi: u32;
        let mut ndlo: u32;
        let mut i: u32;
        let mut e: i32;
        let mut ndebias: i32;
        let mut prefix = 0u8;
        if hi(t) & 0x80000000 != 0 {
            prefix = b'-';
        } else if sf & STRFMT_F_PLUS != 0 {
            prefix = b'+';
        } else if sf & STRFMT_F_SPACE != 0 {
            prefix = b' ';
        }
        prec = prec.wrapping_add((((prec as i32) >> 31) & 7) as u32); // Default precision is 6.
        if strfmt_fp(sf) == strfmt_fp(STRFMT_T_FP_G) {
            // %g - decrement precision if non-zero (to make it like %e).
            prec = prec.wrapping_sub(1);
            prec ^= ((prec as i32) >> 31) as u32;
        }
        // Upstream's `goto rescale_failed` re-enters the conversion without rescaling.
        let mut rescale = sf & STRFMT_T_FP_E != 0 && prec < 14 && n != 0.0;
        'convert: loop {
            t = n.to_bits();
            e = ((hi(t) >> 20) & 0x7ff) as i32;
            ndhi = 0;
            ndebias = 0;
            let mut load_t_lo = false;
            if rescale {
                // Precision is sufficiently low that rescaling will probably work.
                ndebias = RESCALE_E[(e >> 6) as usize] as i32;
                if ndebias != 0 {
                    let mut tn = n * RESCALE_N[(e >> 6) as usize];
                    if e == 0 {
                        tn *= 1e10;
                        ndebias -= 10;
                    }
                    t = tn.to_bits().wrapping_sub(2); // Convert 2ulp below (later we convert 2ulp above).
                    nd[0] = 0x100000 | (hi(t) & 0xfffff);
                    e = ((hi(t) >> 20) & 0x7ff) as i32 - 1075;
                    load_t_lo = true;
                }
            }
            if !load_t_lo {
                nd[0] = hi(t) & 0xfffff;
                if e == 0 {
                    e += 1;
                } else {
                    nd[0] |= 0x100000;
                }
                e -= 1043;
                if lo(t) != 0 {
                    e -= 32;
                    load_t_lo = true;
                }
            }
            if load_t_lo {
                nd[0] = (nd[0] << 3) | (lo(t) >> 29);
                ndhi = nd_mul2k(&mut nd, ndhi, 29, lo(t) & 0x1fffffff, sf);
            }
            if e >= 0 {
                ndhi = nd_mul2k(&mut nd, ndhi, e as u32, 0, sf);
                ndlo = 0;
            } else if sf & STRFMT_T_FP_E == 0 && e < -500 {
                // Below 2^-447, so zero at any %f precision; dividing would overrun the ring.
                nd[0] = 0;
                ndhi = 0;
                ndlo = 0;
            } else {
                ndlo = nd_div2k(&mut nd, ndhi, (-e) as u32, sf);
                if ndhi != 0 && nd[ndhi as usize] == 0 {
                    ndhi -= 1;
                }
            }
            // abs(n) == nd * 10^ndebias (for slightly loose interpretation of ==)
            let mut g_format_like_f = false;
            if sf & STRFMT_T_FP_E != 0 {
                // %e or %g - assume %e and start by calculating nd's exponent (nde).
                let mut eprefix = b'+';
                let mut nde: i32 = -1;
                if ndlo != 0 && nd[ndhi as usize] == 0 {
                    ndhi = 64;
                    loop {
                        ndhi -= 1;
                        if nd[ndhi as usize] != 0 {
                            break;
                        }
                    }
                    nde -= 64 * 9;
                }
                let hilen = ndigits_dec(nd[ndhi as usize]);
                nde = nde.wrapping_add((ndhi * 9 + hilen) as i32);
                if ndebias != 0 {
                    // Rescaling was performed, but this introduced some error, and might
                    // have pushed us across a rounding boundary. We check whether this
                    // error affected the result by introducing even more error (2ulp in
                    // either direction), and seeing whether a rounding boundary was
                    // crossed. Having already converted the -2ulp case, we save off its
                    // most significant digits, convert the +2ulp case, and compare them.
                    let eidx = e + 70 + (lo(t) >= 0xfffffffe && (!hi(t) << 12) == 0) as i32;
                    debug_assert!((0..128).contains(&eidx), "bad eidx {eidx}");
                    let m_e = &FOUR_ULP_M_E[(eidx * 2) as usize..];
                    nd[33] = nd[ndhi as usize];
                    nd[32] = nd[(ndhi.wrapping_sub(1) & 0x3f) as usize];
                    nd[31] = nd[(ndhi.wrapping_sub(2) & 0x3f) as usize];
                    nd_add_m10e(&mut nd, ndhi, m_e[0] as u8, m_e[1] as i32);
                    if !nd_similar(&nd, ndhi, 33, hilen, prec + 1) {
                        rescale = false;
                        continue 'convert;
                    }
                }
                if (prec.wrapping_sub(nde as u32) as i32)
                    < (0x3f & (ndlo as i32).wrapping_neg()) * 9
                {
                    // Precision is sufficiently low as to maybe require rounding.
                    let at = (nde as u32).wrapping_sub(prec).wrapping_sub(1) as i32;
                    // A rescaled number can't be an exact tie at these precisions.
                    let tail_zero = ndebias == 0 && multiple_of_pow10(n, at);
                    ndhi = nd_round(&mut nd, ndhi, at, tail_zero);
                    nde += (hilen != ndigits_dec(nd[ndhi as usize])) as i32;
                }
                nde += ndebias;
                if sf & STRFMT_T_FP_F != 0 {
                    // %g
                    if prec as i32 >= nde && nde >= -4 {
                        if nde < 0 {
                            ndhi = 0;
                        }
                        prec = prec.wrapping_sub(nde as u32);
                        g_format_like_f = true;
                    } else if sf & STRFMT_F_ALT == 0 && prec != 0 && width > 5 {
                        // Decrease precision in order to strip trailing zeroes.
                        let mut tail = [0u8; 9];
                        let maxprec = hilen - 1 + (ndhi.wrapping_sub(ndlo) & 0x3f) * 9;
                        if prec >= maxprec {
                            prec = maxprec;
                        } else {
                            ndlo = ndhi
                                .wrapping_sub(((prec.wrapping_sub(hilen) as i32 + 9) / 9) as u32)
                                & 0x3f;
                        }
                        i = prec
                            .wrapping_sub(hilen)
                            .wrapping_sub((ndhi.wrapping_sub(ndlo) & 0x3f) * 9)
                            .wrapping_add(10);
                        wuint9(&mut tail, 0, nd[ndlo as usize]);
                        while prec != 0 && {
                            i -= 1;
                            tail[i as usize] == b'0'
                        } {
                            prec -= 1;
                            if i == 0 {
                                if ndlo == ndhi {
                                    prec = 0;
                                    break;
                                }
                                ndlo = (ndlo + 1) & 0x3f;
                                wuint9(&mut tail, 0, nd[ndlo as usize]);
                                i = 9;
                            }
                        }
                    }
                }
                if !g_format_like_f {
                    if nde < 0 {
                        // Make nde non-negative.
                        eprefix = b'-';
                        nde = -nde;
                    }
                    len = 3
                        + prec
                        + (prefix != 0) as u32
                        + ndigits_dec(nde as u32)
                        + (nde < 10) as u32
                        + ((prec | (sf & STRFMT_F_ALT)) != 0) as u32;
                    p = buf_more(sb, width.max(len) + 5);
                    if sf & (STRFMT_F_LEFT | STRFMT_F_ZERO) == 0 {
                        while width > len {
                            width -= 1;
                            sb[p] = b' ';
                            p += 1;
                        }
                    }
                    if prefix != 0 {
                        sb[p] = prefix;
                        p += 1;
                    }
                    if sf & (STRFMT_F_LEFT | STRFMT_F_ZERO) == STRFMT_F_ZERO {
                        while width > len {
                            width -= 1;
                            sb[p] = b'0';
                            p += 1;
                        }
                    }
                    let q = wint(sb, p + 1, nd[ndhi as usize] as i32);
                    sb[p] = sb[p + 1]; // Put leading digit in the correct place.
                    if (prec | (sf & STRFMT_F_ALT)) != 0 {
                        // Emit fractional part.
                        sb[p + 1] = b'.';
                        p += 2;
                        prec = prec.wrapping_sub((q - p) as u32); // Account for digits already emitted.
                        p = q;
                        // Then emit chunks of 9 digits (this may emit 8 digits too many).
                        i = ndhi;
                        while prec as i32 > 0 && i != ndlo {
                            i = i.wrapping_sub(1) & 0x3f;
                            p = wuint9(sb, p, nd[i as usize]);
                            prec = prec.wrapping_sub(9);
                        }
                        if sf & STRFMT_T_FP_F != 0 && sf & STRFMT_F_ALT == 0 {
                            // %g (and not %#g) - strip trailing zeroes.
                            p = p.wrapping_add_signed(
                                ((prec as i32) & ((prec as i32) >> 31)) as isize,
                            );
                            while sb[p - 1] == b'0' {
                                p -= 1;
                            }
                            if sb[p - 1] == b'.' {
                                p -= 1;
                            }
                        } else {
                            // %e (or %#g) - emit trailing zeroes.
                            while prec as i32 > 0 {
                                sb[p] = b'0';
                                p += 1;
                                prec -= 1;
                            }
                            p = p.wrapping_add_signed(prec as i32 as isize);
                        }
                    } else {
                        p += 1;
                    }
                    sb[p] = if sf & STRFMT_F_UPPER != 0 { b'E' } else { b'e' };
                    sb[p + 1] = eprefix; // + or -
                    p += 2;
                    if nde < 10 {
                        sb[p] = b'0'; // Always at least two digits of exponent.
                        p += 1;
                    }
                    p = wint(sb, p, nde);
                    break 'convert;
                }
            } else if prec < (0x3f & (ndlo as i32).wrapping_neg()) as u32 * 9 {
                // %f: precision is sufficiently low as to maybe require rounding.
                let at = 0u32.wrapping_sub(prec).wrapping_sub(1) as i32;
                ndhi = nd_round(&mut nd, ndhi, at, multiple_of_pow10(n, at));
            }
            // g_format_like_f:
            if sf & STRFMT_T_FP_E != 0 && sf & STRFMT_F_ALT == 0 && prec != 0 && width != 0 {
                // Decrease precision in order to strip trailing zeroes.
                if ndlo != 0 {
                    // nd has a fractional part; we need to look at its digits.
                    let mut tail = [0u8; 9];
                    let maxprec = (64 - ndlo) * 9;
                    if prec >= maxprec {
                        prec = maxprec;
                    } else {
                        ndlo = 64 - prec.div_ceil(9);
                    }
                    i = prec.wrapping_sub((63 - ndlo) * 9);
                    wuint9(&mut tail, 0, nd[ndlo as usize]);
                    while prec != 0 && {
                        i -= 1;
                        tail[i as usize] == b'0'
                    } {
                        prec -= 1;
                        if i == 0 {
                            if ndlo == 63 {
                                prec = 0;
                                break;
                            }
                            ndlo += 1;
                            wuint9(&mut tail, 0, nd[ndlo as usize]);
                            i = 9;
                        }
                    }
                } else {
                    // nd has no fractional part, so precision goes straight to zero.
                    prec = 0;
                }
            }
            len = ndhi * 9
                + ndigits_dec(nd[ndhi as usize])
                + prec
                + (prefix != 0) as u32
                + ((prec | (sf & STRFMT_F_ALT)) != 0) as u32;
            p = buf_more(sb, width.max(len) + 8);
            if sf & (STRFMT_F_LEFT | STRFMT_F_ZERO) == 0 {
                while width > len {
                    width -= 1;
                    sb[p] = b' ';
                    p += 1;
                }
            }
            if prefix != 0 {
                sb[p] = prefix;
                p += 1;
            }
            if sf & (STRFMT_F_LEFT | STRFMT_F_ZERO) == STRFMT_F_ZERO {
                while width > len {
                    width -= 1;
                    sb[p] = b'0';
                    p += 1;
                }
            }
            // Emit integer part.
            p = wint(sb, p, nd[ndhi as usize] as i32);
            i = ndhi;
            while i != 0 {
                i -= 1;
                p = wuint9(sb, p, nd[i as usize]);
            }
            if (prec | (sf & STRFMT_F_ALT)) != 0 {
                // Emit fractional part.
                sb[p] = b'.';
                p += 1;
                // Emit chunks of 9 digits (this may emit 8 digits too many).
                while prec as i32 > 0 && i != ndlo {
                    i = i.wrapping_sub(1) & 0x3f;
                    p = wuint9(sb, p, nd[i as usize]);
                    prec = prec.wrapping_sub(9);
                }
                if sf & STRFMT_T_FP_E != 0 && sf & STRFMT_F_ALT == 0 {
                    // %g (and not %#g) - strip trailing zeroes.
                    p = p.wrapping_add_signed(((prec as i32) & ((prec as i32) >> 31)) as isize);
                    while sb[p - 1] == b'0' {
                        p -= 1;
                    }
                    if sb[p - 1] == b'.' {
                        p -= 1;
                    }
                } else {
                    // %f (or %#g) - emit trailing zeroes.
                    while prec as i32 > 0 {
                        sb[p] = b'0';
                        p += 1;
                        prec -= 1;
                    }
                    p = p.wrapping_add_signed(prec as i32 as isize);
                }
            }
            break 'convert;
        }
    }
    if sf & STRFMT_F_LEFT != 0 {
        while width > len {
            width -= 1;
            sb[p] = b' ';
            p += 1;
        }
    }
    p
}
