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
//! - SDP: **supported, and `sdp` is required rather than excluded.**
//!   [`native_lapack`](self) implements every `X*Scalar` trait for
//!   [`MpfrFloat`] — including `xsyevr`, the symmetric eigensolver the
//!   PSD cone projects with — so MPFR serves the SDP path without BLAS.
//!   Those trait *declarations* live behind `#[cfg(feature = "sdp")]`
//!   (`algebra/dense/mod.rs`), so `native_lapack` is gated to match and
//!   the two features are built **together**.
//!
//!   Still `unimplemented!()`: `xgesdd` / `xgesvd` (SVD), reached only
//!   via `solver/chordal/decomp/psd_completion.rs`. Chordal
//!   decomposition is therefore the one SDP path MPFR cannot take yet.
//!
//! # Mutual exclusivity
//!
//! Cannot be combined with `faer-sparse` (faer requires `RealField` on
//! f32/f64) — see the `compile_error!` below, which is the only one in
//! this module.
//!
//! **`sdp` is NOT excluded.** Earlier revisions of this comment said it
//! was, citing a `compile_error!` against `sdp` that does not exist here
//! (the `rational` backend has one; this backend does not) and calling
//! the MPFR-native eigensolver a future item when it was already
//! written. Both readings are wrong and both have cost downstream
//! readers real time — please keep this section true to the `cfg`s
//! directly below rather than to intent.



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
