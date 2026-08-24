#![allow(non_snake_case, missing_docs)]
//! Real-number trait split used in place of `num_traits::Float` /
//! `num_traits::FloatConst` so that non-IEEE backends (exact rational,
//! arbitrary-precision MPFR) can satisfy [`FloatT`].
//!
//! - [`Transcendental`] : sqrt, ln, exp, powf, powi, recip
//! - [`RealConst`]      : the small set of irrational constants the solver
//!                        actually consumes (PI, SQRT_2, FRAC_1_SQRT_2)
//! - [`RealSentinel`]   : `infinity`/`nan`/`is_finite`/`is_nan`/`epsilon`
//!                        plus `min`/`max`. On exact backends, "infinity" is
//!                        a sentinel (no IEEE meaning), `nan` is a sentinel,
//!                        `is_finite` is always true, and `epsilon` is the
//!                        rounding tolerance of the working precision.
//!
//! Blanket impls are provided for every `T: num_traits::Float [+ FloatConst]`,
//! so `f32` and `f64` continue to satisfy `FloatT` with no source change at
//! call sites.

use num_traits::{Float, FloatConst};

/// Transcendental real operations used by the solver.
///
/// Implemented for any `T: num_traits::Float` via a blanket impl, and for
/// non-Float backends (e.g. `RationalReal`, `MpfrFloat`) by hand at the
/// working precision selected for that backend.
pub trait Transcendental: Sized {
    fn sqrt(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn recip(self) -> Self;
    /// sine; only used inside the analytic 3x3 symmetric eigensolver that
    /// pow/exp cones invoke for Hessian factorization.
    fn sin(self) -> Self;
    /// cosine; see [`Transcendental::sin`].
    fn cos(self) -> Self;
    /// two-argument arctangent; see [`Transcendental::sin`].
    fn atan2(self, x: Self) -> Self;
}

impl<T: Float> Transcendental for T {
    #[inline]
    fn sqrt(self) -> Self {
        Float::sqrt(self)
    }
    #[inline]
    fn ln(self) -> Self {
        Float::ln(self)
    }
    #[inline]
    fn exp(self) -> Self {
        Float::exp(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        Float::powf(self, n)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        Float::powi(self, n)
    }
    #[inline]
    fn recip(self) -> Self {
        Float::recip(self)
    }
    #[inline]
    fn sin(self) -> Self {
        Float::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        Float::cos(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        Float::atan2(self, x)
    }
}

/// Irrational/transcendental constants used by the solver.
///
/// The solver only references three: PI (exp cone barrier check),
/// SQRT_2 and FRAC_1_SQRT_2 (PSD triangle packing/unpacking).
/// Backends with run-time precision should return values rounded to the
/// current working precision.
pub trait RealConst: Sized {
    fn PI() -> Self;
    fn SQRT_2() -> Self;
    fn FRAC_1_SQRT_2() -> Self;
}

impl<T: FloatConst> RealConst for T {
    #[inline]
    fn PI() -> Self {
        <T as FloatConst>::PI()
    }
    #[inline]
    fn SQRT_2() -> Self {
        <T as FloatConst>::SQRT_2()
    }
    #[inline]
    fn FRAC_1_SQRT_2() -> Self {
        <T as FloatConst>::FRAC_1_SQRT_2()
    }
}

/// IEEE-style sentinel and predicate operations.
///
/// On IEEE floats these forward to `Float::infinity`/`Float::nan`/etc.
/// On exact backends these return *sentinels*: a value tagged so that
/// arithmetic and comparison still work in a sensible way for the
/// solver's use of these values (typically as upper bounds in
/// minima/maxima or as "no value yet" placeholders).
pub trait RealSentinel: Sized {
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn nan() -> Self;
    fn epsilon() -> Self;
    fn max_value() -> Self;
    fn min_value() -> Self;
    fn is_nan(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    /// Sign-bit predicate: true for negative numbers and IEEE -0.
    /// On non-IEEE backends without signed zero, equivalent to `self < 0`.
    fn is_sign_negative(self) -> bool;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
}

impl<T: Float> RealSentinel for T {
    #[inline]
    fn infinity() -> Self {
        <T as Float>::infinity()
    }
    #[inline]
    fn neg_infinity() -> Self {
        <T as Float>::neg_infinity()
    }
    #[inline]
    fn nan() -> Self {
        <T as Float>::nan()
    }
    #[inline]
    fn epsilon() -> Self {
        <T as Float>::epsilon()
    }
    #[inline]
    fn max_value() -> Self {
        <T as Float>::max_value()
    }
    #[inline]
    fn min_value() -> Self {
        <T as Float>::min_value()
    }
    #[inline]
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
    #[inline]
    fn is_finite(self) -> bool {
        Float::is_finite(self)
    }
    #[inline]
    fn is_infinite(self) -> bool {
        Float::is_infinite(self)
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        Float::is_sign_negative(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        Float::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        Float::max(self, other)
    }
}

/// Diagnostic helper exposing the bit width of an arbitrary-precision
/// scalar's internal representation. Used by the solver to record a
/// per-iteration trace of denominator/mantissa growth (`info.iter_diagnostics`)
/// when running on backends like `RationalReal` or a future MPFR float.
///
/// Default behaviour for IEEE floats: returns `(0, 0)`. The IEEE bit
/// width is fixed (52 mantissa bits for `f64`) and uninteresting
/// per-iteration; the diagnostic only adds value when the underlying
/// representation has variable bit width, in which case the type
/// provides its own impl returning meaningful values.
pub trait BitWidthDiagnostic {
    /// Returns `(numer_bits, denom_bits)` for arbitrary-precision
    /// rationals (interpreted loosely: `numer_bits` is "value" and
    /// `denom_bits` is "scale"; an MPFR float would map mantissa/
    /// exponent into this shape). Returns `(0, 0)` for fixed-width
    /// IEEE floats — the default.
    fn bit_width(&self) -> (u64, u64) {
        (0, 0)
    }
}

impl<T: Float> BitWidthDiagnostic for T {}

