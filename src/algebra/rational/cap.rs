//! `maybe_cap` — opt-in inner-loop precision capping.
//!
//! See `precision::set_max_arena_bits` for the public knob. This module
//! holds the helper applied inside every arithmetic op on
//! `RationalReal` to keep BigRational coefficient growth bounded when
//! the cap is engaged.

use super::precision::max_arena_bits;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed};

/// Round `r = n/d` to `m / 2^p` if either `|n|.bits() > p` or
/// `d.bits() > p`. Otherwise leave `r` unchanged. `p` is the current
/// thread-local `max_arena_bits()`; if that is `None`, returns `r`
/// unchanged (exact mode).
///
/// Rounding rule: round-to-nearest, ties away from zero. Error is
/// at most one ulp at precision `p`, i.e. ≤ 2^-p.
#[inline]
pub(crate) fn maybe_cap(r: BigRational) -> BigRational {
    let p = match max_arena_bits() {
        None => return r,
        Some(p) => p,
    };
    let p64 = u64::from(p);
    let needs_round = r.numer().bits() > p64 || r.denom().bits() > p64;
    if !needs_round {
        return r;
    }
    round_to_pow2_denominator(&r, p)
}

/// Round `r = n/d` to `m / 2^p`, regardless of whether the input would
/// already fit. Used both by `maybe_cap` (when the input exceeds the
/// cap) and by the transcendental module (which always rounds to a
/// fixed precision).
pub(crate) fn round_to_pow2_denominator(r: &BigRational, p: u32) -> BigRational {
    let two_p: BigInt = BigInt::one() << (p as usize);
    let scaled = r.numer() * &two_p;
    let d = r.denom();
    // Round to nearest, ties away from zero: m = (scaled + sign(scaled) * d/2) / d
    let half: BigInt = d / 2;
    let rounded = if scaled.is_negative() {
        (scaled - half) / d
    } else {
        (scaled + half) / d
    };
    BigRational::new(rounded, two_p)
}
