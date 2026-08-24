//! Exact-rational backend (`bigrational` feature).
//!
//! This module provides [`RationalReal`], a scalar type satisfying
//! [`FloatT`](crate::algebra::FloatT) whose arithmetic is backed by
//! arbitrary-precision `num_rational::BigRational` stored in a per-thread
//! arena. LP/QP iterates are bit-exact rationals; SOCP/exp/pow barrier
//! transcendentals (`sqrt`, `ln`, `exp`, `powf`) are computed at a
//! configurable thread-local working precision via Newton/Taylor series.
//!
//! # Memory model
//!
//! `RationalReal` is a `Copy`-able 4-byte handle (`u32`) into the arena.
//! Every arithmetic operation appends a fresh `BigRational` and returns a
//! new handle. Memory grows monotonically during a solve. Call
//! [`reset_arena`] between solves to reclaim it; all outstanding
//! `RationalReal` handles invalidate at that point.
//!
//! # Send + Sync
//!
//! [`FloatT`](crate::algebra::FloatT) requires `Send + Sync`. The arena is
//! per-thread, so a `RationalReal` produced on thread A is meaningless on
//! thread B. We assert this with `unsafe impl Send + Sync` on
//! `RationalReal`, with the documented invariant: **a `RationalReal` may
//! only be dereferenced on the thread that produced it.** Within a single
//! `solver.solve()` call (Clarabel's IPM is single-threaded for a given
//! problem) this holds. Outer parallelism — running independent solves on
//! independent threads, each with its own arena — is fully supported.
//!
//! Sending a `RationalReal` across threads in user code is a logic bug
//! that this module cannot detect; users should not do it. The intended
//! usage is: instantiate `Solver<RationalReal>`, run `solve()`, extract
//! results via [`RationalReal::to_bigrational`] or
//! [`RationalReal::into_pair`] before any thread-crossing.
//!
//! # Mutual exclusivity
//!
//! Cannot be combined with `sdp` / `sdp-*` or `faer-sparse`. Those
//! features pin `T` to `f32`/`f64` for BLAS / `faer::RealField`
//! operations. The combination is rejected at compile time via the
//! `compile_error!`s below.
//!
//! **For SDP work**: this backend is the wrong tool. PSD cone
//! projection requires eigendecomposition, which is not exact in
//! rationals (eigenvalues of rational symmetric matrices are
//! algebraic numbers, not generally rational). Two production paths:
//!
//! - **High-precision SDP**: use the [`mpfr`](crate::algebra::mpfr)
//!   backend (`MpfrFloat`) — bounded denominator size, configurable
//!   precision, supports the existing BLAS path.
//!
//! - **Certified rational SDP solutions**: solve in `f64` or
//!   `MpfrFloat`, then round the dual back to exact rationals via a
//!   Peyrl–Parrilo-style tightening step. The standard recipe is
//!   continued-fraction rational rounding + an exact-arithmetic
//!   feasibility verify in `RationalReal`. Helpers for this are provided
//!   as [`tighten_scalar`], [`tighten_vec`], and [`tighten_psd_block`].

#[cfg(feature = "sdp")]
compile_error!(
    "the `bigrational` feature is mutually exclusive with `sdp` and `sdp-*` \
     because SDP requires BLAS/LAPACK on f32/f64"
);

#[cfg(feature = "faer-sparse")]
compile_error!(
    "the `bigrational` feature is mutually exclusive with `faer-sparse` \
     because faer requires `RealField` on f32/f64"
);

mod arena;
mod cap;
mod display;
mod precision;
mod real;
mod sentinel;
mod tighten;
mod transcendental;
#[cfg(feature = "serde")]
mod serde_impl;
#[cfg(feature = "rug-interop")]
mod rug_interop;

pub use arena::{arena_len, reset_arena};
pub use precision::{
    max_arena_bits, precision_bits, set_max_arena_bits, set_precision_bits, with_max_arena_bits,
    with_precision,
};
pub use real::RationalReal;
pub use tighten::{tighten_psd_block, tighten_scalar, tighten_vec};

// Compile-time assertion: RationalReal satisfies CoreFloatT and (because
// neither sdp nor faer-sparse can be enabled with bigrational) FloatT.
#[allow(dead_code)]
fn _assert_rational_real_is_floatt() {
    fn assert_floatt<T: crate::algebra::FloatT>() {}
    assert_floatt::<RationalReal>();
}

#[cfg(test)]
mod tests;
