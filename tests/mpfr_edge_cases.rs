//! Public-API edge cases for the MPFR backend.
//!
//! This file is an *integration* test: it is compiled as a separate crate
//! and can therefore only use `clarabel`'s **public** API.
//!
//! It used to also contain three `xsyevr` / `xpotrf` / `xsyrk` cases that
//! named the `BlasFloatT` trait. That trait is declared in
//! `src/algebra/dense/blas/traits.rs`, and `dense` is a *private* module of
//! `crate::algebra` re-exported only as `pub(crate) use dense::*`, so no
//! path outside the crate can reach it. The file failed to build with
//! 3 x `E0405: cannot find trait BlasFloatT in this scope`, and because a
//! broken test target fails the whole `cargo test` build, it made the entire
//! mpfr suite unreachable unless `--lib` was passed. Those three cases now
//! live in-crate, in `src/algebra/mpfr/tests.rs::lapack_tests`.
//!
//! What remains here is worth keeping as an integration test precisely
//! because it is public-API-only: it is the check that `MpfrFloat` and the
//! `RealSentinel` trait really are reachable from a downstream crate.

#![cfg(all(feature = "sdp", feature = "mpfr"))]

use clarabel::algebra::*;

fn to_mpfr(v: f64) -> MpfrFloat {
    MpfrFloat::from(v)
}

#[test]
fn test_mpfr_sentinels() {
    let inf = <MpfrFloat as RealSentinel>::infinity();
    let neg_inf = <MpfrFloat as RealSentinel>::neg_infinity();
    let nan = <MpfrFloat as RealSentinel>::nan();

    assert!(inf > to_mpfr(1e100));
    assert!(neg_inf < to_mpfr(-1e100));
    assert!(<MpfrFloat as RealSentinel>::is_nan(nan));
    assert!(<MpfrFloat as RealSentinel>::is_infinite(inf));
    assert!(!<MpfrFloat as RealSentinel>::is_finite(nan));
    assert!(<MpfrFloat as RealSentinel>::is_finite(to_mpfr(0.0)));
}
