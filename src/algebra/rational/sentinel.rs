//! [`RealSentinel`] and [`RealConst`] for [`RationalReal`].
//!
//! On IEEE floats these correspond to actual hardware semantics
//! (`f64::INFINITY`, `f64::NAN`, `Float::is_finite`). On the rational
//! backend we encode IEEE-style sentinels in the **top two bits** of the
//! `u32` handle:
//!
//! - `infinity()` → the `POS_INF_HANDLE` constant (tag = `0b01`,
//!   index part unused; no arena entry).
//! - `neg_infinity()` → `NEG_INF_HANDLE` (tag = `0b10`).
//! - `nan()` → `NAN_HANDLE` (tag = `0b11`).
//! - `epsilon()` → `2⁻ᵖ` where p is the current thread-local working
//!   precision (used as the rounding tolerance for transcendentals); a
//!   regular finite-tagged handle pushed into the arena.
//! - `min`/`max` → IEEE-style: any-NaN ⇒ NaN; otherwise the
//!   smaller/larger of the two operands.
//!
//! Because the sentinels are pure tag bits with no associated arena
//! entry, they are **immune to arena resets**: `infinity()` returns the
//! same constant before and after `reset_arena()`. Cross-thread send
//! is still forbidden for finite handles (their index references the
//! per-thread arena), but sentinel handles are thread-independent.

use super::arena;
use super::precision::precision_bits;
use super::real::RationalReal;
use crate::algebra::transcendental::{RealConst, RealSentinel};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One};
use std::cell::Cell;

impl RealSentinel for RationalReal {
    fn infinity() -> Self {
        RationalReal(arena::POS_INF_HANDLE)
    }

    fn neg_infinity() -> Self {
        RationalReal(arena::NEG_INF_HANDLE)
    }

    fn nan() -> Self {
        RationalReal(arena::NAN_HANDLE)
    }

    /// `2^-p` where p is the current thread-local working precision in
    /// bits. Used by the solver as a rounding tolerance for
    /// transcendentals; basic arithmetic on `RationalReal` is exact
    /// regardless of this value.
    fn epsilon() -> Self {
        let p = precision_bits();
        let denom = BigInt::from(1u8) << (p as usize);
        Self::from_pair(BigInt::one(), denom)
    }

    /// Sentinel for "very large positive" — same as `infinity()` here.
    fn max_value() -> Self {
        Self::infinity()
    }

    /// Sentinel for "very small positive" — `2^-p` (same as `epsilon`).
    fn min_value() -> Self {
        Self::epsilon()
    }

    #[inline]
    fn is_nan(self) -> bool {
        arena::is_nan_handle(self.0)
    }

    #[inline]
    fn is_finite(self) -> bool {
        arena::is_finite_handle(self.0)
    }

    #[inline]
    fn is_infinite(self) -> bool {
        arena::is_infinite_handle(self.0)
    }

    fn is_sign_negative(self) -> bool {
        // IEEE: NaN is not classified as negative; -inf is. For finite
        // values, equivalent to self < 0 (no signed zero in rationals).
        if arena::is_nan_handle(self.0) {
            return false;
        }
        if arena::is_neg_inf_handle(self.0) {
            return true;
        }
        if arena::is_pos_inf_handle(self.0) {
            return false;
        }
        arena::with(self.0, |a| {
            use num_traits::Signed;
            a.is_negative()
        })
    }

    /// IEEE-style `min`: NaN-poisoned (NaN ⇒ NaN), otherwise returns the
    /// smaller of the two values. Note that this differs from the
    /// "NaN-suppressing" `f64::min` and matches `f64::minimum`.
    fn min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan();
        }
        if self <= other {
            self
        } else {
            other
        }
    }

    /// IEEE-style `max`: NaN-poisoned. See [`Self::min`].
    fn max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::nan();
        }
        if self >= other {
            self
        } else {
            other
        }
    }
}

// ============================================================
// RealConst — irrational constants, computed once at the working
// precision and cached per-thread.
// ============================================================

thread_local! {
    /// Per-thread cache of the irrational constants. Invalidated when
    /// the working precision changes OR the arena generation changes
    /// (reset_arena()).
    static CONST_CACHE: Cell<Option<ConstCache>> = const { Cell::new(None) };
}

#[derive(Copy, Clone)]
struct ConstCache {
    bits: u32,
    generation: u64,
    pi: u32,
    sqrt_2: u32,
    frac_1_sqrt_2: u32,
}

fn const_cache() -> ConstCache {
    let bits = precision_bits();
    let gen = arena::arena_generation();
    if let Some(c) = CONST_CACHE.with(|c| c.get()) {
        if c.bits == bits && c.generation == gen {
            return c;
        }
    }
    // Compute fresh at current precision.
    let sqrt_2 = super::transcendental::sqrt_newton_pub(
        &BigRational::from_i32(2).unwrap(),
        bits,
    );
    let frac_1_sqrt_2 = BigRational::one() / &sqrt_2;
    // π via Machin's formula: π/4 = 4·arctan(1/5) - arctan(1/239),
    // using arctan(z) = z - z³/3 + z⁵/5 - … (alternating series; |z|<1).
    let pi = machin_pi(bits);
    let cache = ConstCache {
        bits,
        generation: gen,
        pi: arena::push(pi),
        sqrt_2: arena::push(sqrt_2),
        frac_1_sqrt_2: arena::push(frac_1_sqrt_2),
    };
    CONST_CACHE.with(|c| c.set(Some(cache)));
    cache
}

/// arctan(1/n) via the alternating Taylor series. Truncates and rounds
/// to bounded precision after each term so BigRational coefficients
/// stay capped at `p+16` bits.
fn arctan_recip(n: u32, p: u32) -> BigRational {
    use super::cap::round_to_pow2_denominator;
    use num_traits::Signed;
    let n_r = BigRational::from_u32(n).unwrap();
    let z = BigRational::one() / &n_r; // 1/n
    let z2 = &z * &z;
    let mut term = z.clone();
    let mut sum = z;
    let tol = BigRational::new(BigInt::one(), BigInt::one() << ((p + 4) as usize));
    let mut k: u64 = 1;
    let mut sign = -1i32;
    loop {
        // term ← term · z²
        term = round_to_pow2_denominator(&(&term * &z2), p + 16);
        k += 2;
        let denom = BigRational::from_u64(k).unwrap();
        let add = &term / &denom;
        if sign > 0 {
            sum = round_to_pow2_denominator(&(&sum + &add), p + 16);
        } else {
            sum = round_to_pow2_denominator(&(&sum - &add), p + 16);
        }
        sign = -sign;
        if add.abs() < tol {
            break;
        }
        if k > 50_000 {
            break; // safety
        }
    }
    sum
}

/// π via Machin's formula at `p` bits of fractional precision.
fn machin_pi(p: u32) -> BigRational {
    use super::cap::round_to_pow2_denominator;
    let four = BigRational::from_i32(4).unwrap();
    let a = arctan_recip(5, p + 8);
    let b = arctan_recip(239, p + 8);
    let pi_over_4 = &four * &a - &b;
    let pi = &pi_over_4 * &four;
    round_to_pow2_denominator(&pi, p)
}

#[allow(non_snake_case)]
impl RealConst for RationalReal {
    fn PI() -> Self {
        let c = const_cache();
        RationalReal(c.pi)
    }

    fn SQRT_2() -> Self {
        let c = const_cache();
        RationalReal(c.sqrt_2)
    }

    fn FRAC_1_SQRT_2() -> Self {
        let c = const_cache();
        RationalReal(c.frac_1_sqrt_2)
    }
}
