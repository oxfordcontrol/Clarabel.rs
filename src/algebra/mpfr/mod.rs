//! Run-time MPFR float backend (`mpfr` feature).
//!
//! [`MpfrFloat`] wraps [`rug::Float`] (a thin MPFR `mpfr_t` wrapper)
//! and satisfies [`FloatT`](crate::algebra::FloatT). All `f32`/`f64`
//! IEEE operations have direct MPFR counterparts at configurable
//! precision; the working precision is read at construction time and
//! propagates through arithmetic.
//!
//! # Precision model
//!
//! Each [`MpfrFloat`] carries its own MPFR precision (in bits). Binary
//! ops between values of different precision use the maximum of the
//! two, matching `rug`'s convention. The default precision for new
//! `MpfrFloat` values is set per-thread via [`set_default_precision`]
//! / [`with_precision`]; default is 167 bits ≈ 50 decimal digits,
//! aligning with QOU's R5_FULL_PLAN.md target.
//!
//! # vs. bigrational
//!
//! - Bounded denominator size by construction; no runtime cost from
//!   denominator blow-up. Practical for problems where exact rational
//!   arithmetic is intractable (cf. `examples/lp_rational.rs`).
//! - Not bit-exact: arithmetic ops round to working precision. The
//!   trade-off is "high-precision floats" semantics: ULP at ~50 dps
//!   instead of f64's 16 dps, vs. truly exact rationals.
//! - SDP integration: still excluded by the `compile_error!` against
//!   `sdp` (BLAS/LAPACK only impl on f32/f64). A future MPFR-native
//!   eigensolver would unlock that.
//!
//! # Mutual exclusivity
//!
//! Cannot be combined with `sdp` or `faer-sparse`.



#[cfg(feature = "faer-sparse")]
compile_error!(
    "the `mpfr` feature is mutually exclusive with `faer-sparse` \
     because faer requires `RealField` on f32/f64"
);

mod precision;
mod real;
mod transcendental;
mod arena;
mod sentinel;
// `native_lapack` implements the X*Scalar traits, which `algebra/dense/mod.rs`
// declares only under `#[cfg(feature = "sdp")]`. Declared unconditionally, it
// makes `--features mpfr` alone fail to compile (13 x E0405).
#[cfg(feature = "sdp")]
mod native_lapack;
#[cfg(feature = "serde")]
mod serde_impl;

pub use precision::{
    default_precision, set_default_precision, with_precision as with_mpfr_precision,
};
pub use real::MpfrFloat;
pub use arena::{arena_len, reset_arena};

// Compile-time assertion: MpfrFloat satisfies CoreFloatT (and, via the
// vacuous MaybeBlasFloatT/MaybeFaerFloatT bounds when neither sdp nor
// faer-sparse is enabled, FloatT).
#[allow(dead_code)]
fn _assert_mpfr_float_is_floatt() {
    fn assert_floatt<T: crate::algebra::FloatT>() {}
    assert_floatt::<MpfrFloat>();
}

#[cfg(test)]
mod tests;
