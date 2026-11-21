//! [`Transcendental`] impl for [`RationalReal`].
//!
//! - `sqrt`: Newton iteration x ← (x + a/x)/2 with denominator capping.
//! - `recip`: exact `BigRational` reciprocal.
//! - `powi`: exact repeated squaring (no precision loss).
//! - `ln`: argument reduction `ln(a) = k·ln 2 + ln(m)` with m ∈ [1, 2),
//!   then Taylor on `ln((1+y)/(1-y)) = 2·(y + y³/3 + y⁵/5 + …)`.
//! - `exp`: range reduction `exp(x) = 2^k · exp(r)` with `k = floor(x / ln 2)`,
//!   then Taylor on `exp(r) = Σ rⁿ/n!`.
//! - `powf(a, b) = exp(b · ln a)`.
//! - `sin`/`cos`/`atan2`: stubbed `unimplemented!()`; only called by the
//!   analytic 3×3 symmetric eigensolver in `dense3x3/eigen.rs`, which is
//!   used by SDP code paths that the `bigrational` feature excludes via
//!   `compile_error!` in `mod.rs`. They exist as trait methods so the
//!   `Transcendental` impl is complete.
//!
//! All transcendentals round their result to the current thread-local
//! working precision (`precision_bits()`) to keep `BigRational`
//! denominators bounded.

use super::arena;
use super::cap::round_to_pow2_denominator as round_to_precision;
use super::precision::precision_bits;
use super::real::RationalReal;
use crate::algebra::transcendental::Transcendental;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

/// `2^-p` as a [`BigRational`]. Used as the convergence tolerance.
fn tolerance(p: u32) -> BigRational {
    BigRational::new(BigInt::one(), BigInt::one() << (p as usize))
}

/// Two as a [`BigRational`].
fn two() -> BigRational {
    BigRational::from_i32(2).unwrap()
}

// =========================================================
// sqrt: Newton iteration
// =========================================================

/// Public-in-module shim for use by sibling modules
/// (`sentinel.rs::const_cache` for SQRT_2 / FRAC_1_SQRT_2).
#[inline]
pub(crate) fn sqrt_newton_pub(a: &BigRational, p: u32) -> BigRational {
    sqrt_newton(a, p)
}

fn sqrt_newton(a: &BigRational, p: u32) -> BigRational {
    if a.is_zero() {
        return BigRational::zero();
    }
    if a.is_negative() {
        panic!("RationalReal::sqrt of a negative value");
    }
    let tol = tolerance(p);
    // Initial guess: round-trip through f64 — gives 53 bits, Newton
    // doubles precision per iteration so we converge in ~log2(p/53) steps.
    let init = a
        .to_f64()
        .map(|v| v.sqrt())
        .and_then(BigRational::from_f64)
        .unwrap_or_else(BigRational::one);
    let mut x = init;
    let two = two();
    for _ in 0..200 {
        let next = (&x + a / &x) / &two;
        let diff = (&next - &x).abs();
        x = round_to_precision(&next, p + 8);
        if diff < tol {
            break;
        }
    }
    round_to_precision(&x, p)
}

// =========================================================
// ln: argument reduction + Taylor on ln((1+y)/(1-y))
// =========================================================

// Per-thread cache of ln(2) at the current working precision.
// Recomputed only when the precision changes; ln/exp call this on
// every invocation so caching is a substantial win.
thread_local! {
    static LN2_CACHE: std::cell::RefCell<Option<(u32, BigRational)>> =
        const { std::cell::RefCell::new(None) };
}

/// Cached `ln(2)` at the current thread-local precision. Falls back
/// to a fresh Taylor-series evaluation when the precision changes.
fn ln2(p: u32) -> BigRational {
    LN2_CACHE.with(|cell| {
        let mut slot = cell.borrow_mut();
        if let Some((cached_p, ref v)) = *slot {
            if cached_p == p {
                return v.clone();
            }
        }
        let v = ln2_compute(p);
        *slot = Some((p, v.clone()));
        v
    })
}

fn ln2_compute(p: u32) -> BigRational {
    // ln(2) = 2 * (y + y³/3 + y⁵/5 + …) with y = (2-1)/(2+1) = 1/3
    let one = BigRational::one();
    let three = BigRational::from_i32(3).unwrap();
    let y = &one / &three;
    let y2 = &y * &y;
    let mut term = y.clone();
    let mut sum = y.clone();
    let tol = tolerance(p + 4);
    let mut k: u64 = 1;
    loop {
        term = round_to_precision(&(&term * &y2), p + 16);
        k += 2;
        let denom = BigRational::from_u64(k).unwrap();
        let add = &term / &denom;
        sum = round_to_precision(&(&sum + &add), p + 16);
        if add.abs() < tol {
            break;
        }
    }
    round_to_precision(&(&two() * &sum), p)
}

fn ln_rational(a: &BigRational, p: u32) -> BigRational {
    if a <= &BigRational::zero() {
        panic!("RationalReal::ln of a non-positive value");
    }
    // Argument reduction: ln(a) = k·ln 2 + ln(m), m = a / 2^k ∈ [1, 2).
    // Estimate k = floor(log2(a)) ≈ numer.bits() - denom.bits() in
    // O(1) via BigInt::bits, then refine with at most 1-2 ±1 fixups
    // (the bit-length estimate is exact for powers of 2 and at most
    // 1 off otherwise — k is the exponent of the *leading* binary
    // digit so refinement always converges in ≤2 iterations).
    let two_r = two();
    let one_r = BigRational::one();
    let mut k: i64 = (a.numer().bits() as i64) - (a.denom().bits() as i64);
    let mut m = if k > 0 {
        let pow: BigInt = BigInt::one() << (k as usize);
        a / BigRational::from(pow)
    } else if k < 0 {
        let pow: BigInt = BigInt::one() << ((-k) as usize);
        a * BigRational::from(pow)
    } else {
        a.clone()
    };
    while m >= two_r {
        m = m / &two_r;
        k += 1;
    }
    while m < one_r {
        m = m * &two_r;
        k -= 1;
    }
    // Now m ∈ [1, 2). Set y = (m-1)/(m+1), |y| ≤ 1/3.
    let y = (&m - &one_r) / (&m + &one_r);
    let y2 = &y * &y;
    let mut term = y.clone();
    let mut sum = y.clone();
    let tol = tolerance(p + 4);
    let mut kk: u64 = 1;
    loop {
        term = round_to_precision(&(&term * &y2), p + 16);
        kk += 2;
        let denom = BigRational::from_u64(kk).unwrap();
        let add = &term / &denom;
        sum = round_to_precision(&(&sum + &add), p + 16);
        if add.abs() < tol {
            break;
        }
    }
    let ln_m = &two_r * &sum;
    let result = if k == 0 {
        ln_m
    } else {
        let k_r = BigRational::from_i64(k).unwrap();
        ln_m + k_r * ln2(p + 8)
    };
    round_to_precision(&result, p)
}

// =========================================================
// exp: range reduction + Taylor
// =========================================================

fn exp_rational(x: &BigRational, p: u32) -> BigRational {
    // exp(x) = 2^k · exp(r) where k = round(x / ln 2), r = x - k·ln 2 small.
    let ln_2 = ln2(p + 16);
    let k_rat = round_to_precision(&(x / &ln_2), 0); // round to integer (denom = 1)
    // Convert k_rat to integer k. Since round_to_precision with p=0 produces
    // m / 1 with m the rounded integer.
    let k_int = k_rat.numer().clone();
    // For very large |k|, exp(x) overflows or underflows beyond any
    // representable rational at the working precision. Threshold: when
    // |k| > 2^30 the exponent x ≈ k·ln 2 is already ~7×10^8 — well
    // beyond any plausible solver iterate's magnitude. We saturate to a
    // very-large finite value (2^256) for positive k or zero for
    // negative k. The outer `Transcendental::exp` wrapper also handles
    // ±inf inputs via the sentinel tag, so this internal saturation
    // only fires when the caller hands us a finite-tagged but
    // unrealistically large iterate (typically a sign of an unrelated
    // upstream bug in the calling code).
    const K_LIMIT: i64 = 1 << 30;
    let k_i64 = match k_int.to_i64() {
        Some(k) if k.abs() <= K_LIMIT => k,
        _ => {
            return if k_int.is_positive() {
                BigRational::new(BigInt::one() << 256u32, BigInt::one())
            } else {
                BigRational::zero()
            };
        }
    };
    let k_back = BigRational::from(BigInt::from(k_i64));
    let r = x - &k_back * &ln_2;
    // Taylor: exp(r) = Σ rⁿ/n!
    let one_r = BigRational::one();
    let mut term = one_r.clone();
    let mut sum = one_r.clone();
    let tol = tolerance(p + 4);
    let mut n: u64 = 1;
    loop {
        let denom = BigRational::from_u64(n).unwrap();
        term = round_to_precision(&(&term * &r / &denom), p + 16);
        sum = round_to_precision(&(&sum + &term), p + 16);
        if term.abs() < tol {
            break;
        }
        n += 1;
        if n > 500 {
            break; // safety
        }
    }
    // Multiply by 2^k.
    let result = if k_i64 >= 0 {
        let scale: BigInt = BigInt::one() << (k_i64 as usize);
        sum * BigRational::from(scale)
    } else {
        let scale: BigInt = BigInt::one() << ((-k_i64) as usize);
        sum / BigRational::from(scale)
    };
    round_to_precision(&result, p)
}

// =========================================================
// powi: exact repeated squaring
// =========================================================

fn powi_exact(base: &BigRational, n: i32) -> BigRational {
    if n == 0 {
        return BigRational::one();
    }
    let mut result = BigRational::one();
    let mut b = if n > 0 { base.clone() } else { BigRational::one() / base };
    let mut e = n.unsigned_abs();
    while e > 0 {
        if e & 1 == 1 {
            result = result * &b;
        }
        e >>= 1;
        if e > 0 {
            b = &b * &b;
        }
    }
    result
}

// =========================================================
// Transcendental impl
// =========================================================

impl Transcendental for RationalReal {
    fn sqrt(self) -> Self {
        // sqrt propagation: NaN → NaN; +inf → +inf; -inf → NaN; -finite → NaN.
        if self.is_nan_tag() || self.is_neg_inf_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        if self.is_pos_inf_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::infinity();
        }
        // finite. Trap negatives at the sentinel boundary rather than
        // panicking inside sqrt_newton.
        if <Self as num_traits::Signed>::is_negative(&self) {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        let p = precision_bits();
        let v = arena::with(self.0, |a| sqrt_newton(a, p));
        RationalReal::from_bigrational(v)
    }

    fn ln(self) -> Self {
        // ln propagation: NaN → NaN; +inf → +inf; -inf → NaN;
        // ln(non-positive finite) → -inf at 0, NaN below 0.
        if self.is_nan_tag() || self.is_neg_inf_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        if self.is_pos_inf_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::infinity();
        }
        if self.is_zero() {
            return <Self as crate::algebra::transcendental::RealSentinel>::neg_infinity();
        }
        if <Self as num_traits::Signed>::is_negative(&self) {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        let p = precision_bits();
        let v = arena::with(self.0, |a| ln_rational(a, p));
        RationalReal::from_bigrational(v)
    }

    fn exp(self) -> Self {
        // exp propagation: NaN → NaN; +inf → +inf; -inf → 0.
        if self.is_nan_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        if self.is_pos_inf_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::infinity();
        }
        if self.is_neg_inf_tag() {
            return Self::zero();
        }
        let p = precision_bits();
        let v = arena::with(self.0, |a| exp_rational(a, p));
        RationalReal::from_bigrational(v)
    }

    fn powf(self, n: Self) -> Self {
        // a^b = exp(b · ln a). The sentinel-input rules here follow
        // IEEE-754 / C99's pow(): inputs and special cases covered
        // explicitly; everything else delegates through ln/exp, which
        // themselves are now sentinel-aware (see above).
        //
        // Order matters: the IEEE 754 standard pins down two cases that
        // *do not* propagate NaN — pow(x, 0) = 1 for *any* x including
        // NaN/±inf, and pow(1, y) = 1 for *any* y including NaN/±inf.
        // These have to be checked before the NaN short-circuit.
        if n.is_zero() {
            // x^0 = 1 for any x (including NaN, ±inf, and 0; matches IEEE 754 / C99 pow).
            return Self::one();
        }
        // 1^y = 1 for any y (including NaN and ±inf), per IEEE 754.
        if self.is_finite_tag() && arena::with(self.0, |r| r.is_one()) {
            return Self::one();
        }
        if self.is_nan_tag() || n.is_nan_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        // Base = ±inf
        if self.is_pos_inf_tag() {
            // (+inf)^positive = +inf; (+inf)^negative = 0
            return if <Self as num_traits::Signed>::is_positive(&n) {
                <Self as crate::algebra::transcendental::RealSentinel>::infinity()
            } else {
                Self::zero()
            };
        }
        if self.is_neg_inf_tag() {
            // (-inf)^pos integer odd  = -inf; even = +inf; non-integer = NaN
            // (-inf)^neg integer odd  = -0;   even = +0;   non-integer = NaN
            // (-inf)^(+inf) = +inf;  (-inf)^(-inf) = +0    (per IEEE 754)
            //
            // We don't track integer-ness on RationalReal explicitly
            // here, but we can detect it: integer means denom == 1.
            let n_is_pos = <Self as num_traits::Signed>::is_positive(&n);
            if !n.is_finite_tag() {
                // n is ±inf (NaN already filtered above). |-inf| > 1 so
                // (-inf)^(+inf) → +inf, (-inf)^(-inf) → +0.
                return if n_is_pos {
                    <Self as crate::algebra::transcendental::RealSentinel>::infinity()
                } else {
                    Self::zero()
                };
            }
            // finite n
            let denom_is_one = arena::with(n.0, |r| r.denom().is_one());
            if !denom_is_one {
                return <Self as crate::algebra::transcendental::RealSentinel>::nan();
            }
            let numer_is_odd = arena::with(n.0, |r| r.numer().bit(0));
            return match (n_is_pos, numer_is_odd) {
                (true, true) => <Self as crate::algebra::transcendental::RealSentinel>::neg_infinity(),
                (true, false) => <Self as crate::algebra::transcendental::RealSentinel>::infinity(),
                // negative integer exponent: result has magnitude 0.
                // Sign collapses to 0 (no signed zero in rationals).
                (false, _) => Self::zero(),
            };
        }
        // Exponent = ±inf, base finite (the |base| == 1 case is
        // handled above by the early `1^y = 1` path; -1^y reaches
        // here, and IEEE-754 says pow(-1, ±inf) = 1 as well).
        if n.is_pos_inf_tag() {
            // |a| > 1 → +inf;  |a| == 1 → 1;  |a| < 1 → 0
            let abs = <Self as num_traits::Signed>::abs(&self);
            let one = Self::one();
            return if abs > one {
                <Self as crate::algebra::transcendental::RealSentinel>::infinity()
            } else if abs == one {
                Self::one()
            } else {
                Self::zero()
            };
        }
        if n.is_neg_inf_tag() {
            let abs = <Self as num_traits::Signed>::abs(&self);
            let one = Self::one();
            return if abs > one {
                Self::zero()
            } else if abs == one {
                Self::one()
            } else {
                <Self as crate::algebra::transcendental::RealSentinel>::infinity()
            };
        }
        // finite^finite path: base = 0 special-cased; other negative-base cases
        // panic via ln (kept as the existing behaviour: solver call sites
        // never produce negative bases inside the barrier evaluation).
        if self.is_zero() {
            return if <Self as num_traits::Signed>::is_positive(&n) {
                Self::zero()
            } else {
                <Self as crate::algebra::transcendental::RealSentinel>::nan()
            };
        }
        let p = precision_bits();
        let v = arena::with2(self.0, n.0, |a, b| {
            let log_a = ln_rational(a, p + 8);
            let prod = round_to_precision(&(b * &log_a), p + 8);
            exp_rational(&prod, p)
        });
        RationalReal::from_bigrational(v)
    }

    fn powi(self, n: i32) -> Self {
        // (NaN)^n = NaN; (±inf)^n: see powf-style rules but n is always integer.
        if self.is_nan_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        if n == 0 {
            return Self::one();
        }
        if self.is_pos_inf_tag() {
            return if n > 0 {
                <Self as crate::algebra::transcendental::RealSentinel>::infinity()
            } else {
                Self::zero()
            };
        }
        if self.is_neg_inf_tag() {
            let odd = (n & 1) != 0;
            return if n > 0 {
                if odd {
                    <Self as crate::algebra::transcendental::RealSentinel>::neg_infinity()
                } else {
                    <Self as crate::algebra::transcendental::RealSentinel>::infinity()
                }
            } else {
                Self::zero()
            };
        }
        // finite. powi_exact divides by `base` for n < 0 and would panic
        // on division by zero — short-circuit that to ±inf via the sentinel.
        if self.is_zero() && n < 0 {
            return <Self as crate::algebra::transcendental::RealSentinel>::infinity();
        }
        let v = arena::with(self.0, |a| powi_exact(a, n));
        RationalReal::from_bigrational(v)
    }

    fn recip(self) -> Self {
        // 1 / NaN = NaN; 1 / ±inf = 0; 1 / 0 = +inf (matches IEEE).
        if self.is_nan_tag() {
            return <Self as crate::algebra::transcendental::RealSentinel>::nan();
        }
        if self.is_infinite_tag() {
            return Self::zero();
        }
        if self.is_zero() {
            return <Self as crate::algebra::transcendental::RealSentinel>::infinity();
        }
        let v = arena::with(self.0, |a| BigRational::one() / a);
        RationalReal::from_bigrational(v)
    }

    fn sin(self) -> Self {
        unimplemented!(
            "RationalReal::sin is not implemented; this method is only \
             reached from the analytic 3x3 symmetric eigensolver in \
             dense3x3/eigen.rs, which is used by SDP code paths that \
             the bigrational feature excludes."
        )
    }

    fn cos(self) -> Self {
        unimplemented!(
            "RationalReal::cos is not implemented; see the comment on \
             RationalReal::sin."
        )
    }

    fn atan2(self, _x: Self) -> Self {
        unimplemented!(
            "RationalReal::atan2 is not implemented; see the comment on \
             RationalReal::sin."
        )
    }
}
