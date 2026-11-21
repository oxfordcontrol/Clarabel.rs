//! Peyrl–Parrilo-style rational tightening.
//!
//! Bridges the f64 / MPFR solver paths to the bigrational backend's
//! exact-arithmetic guarantees. Given a numeric solution `x ∈ ℝⁿ`
//! and a denominator-size budget `Q`, returns the closest rational
//! vector `x' ∈ ℚⁿ` with denominators ≤ `Q` such that:
//!
//! 1. Each component `x'_i` is the [Stern–Brocot continued-fraction]
//!    best rational approximation of `x_i` with `denom(x'_i) ≤ Q`.
//! 2. The resulting vector is returned as `Vec<RationalReal>` for
//!    downstream verification (e.g. exact-arithmetic recomputation
//!    of `Ax' + s' = b` and per-cone membership checks).
//!
//! [Stern–Brocot continued-fraction]: https://en.wikipedia.org/wiki/Stern%E2%80%93Brocot_tree
//!
//! # Caller's responsibility
//!
//! This module **does not** verify cone feasibility of the rounded
//! vector — that's the caller's call (and depends on which cones
//! the user can recompute exactly). The standard recipe for a
//! certified rational SDP solution:
//!
//! 1. Solve in `f64` or `MpfrFloat`.
//! 2. Use [`tighten_to_rational`] to round each entry of the dual `z`
//!    block-by-block to `Q`-bounded rationals.
//! 3. Recompute the per-cone PSD/PSD-residual in `RationalReal`
//!    arithmetic via [`DefaultSolution::dual_psd_block`] and
//!    [`Self::tighten_psd_block`] (separate helper).
//! 4. Verify each PSD block's eigenvalues are non-negative via Sturm
//!    sequence on the characteristic polynomial (out-of-scope here;
//!    QOU has this in their hecke-engine `sturm.rs`).
//!
//! # API shape
//!
//! - [`tighten_scalar`]: single `f64 -> RationalReal` with denominator bound.
//! - [`tighten_vec`]: element-wise over a slice.

use super::real::RationalReal;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, Zero};

/// Round a single `f64` to the closest rational `p/q` with `q ≤ q_max`,
/// using continued-fraction expansion (Stern–Brocot tree).
///
/// Specifically: compute the continued-fraction representation of `x`,
/// truncate at the largest convergent whose denominator is ≤ `q_max`,
/// and return that convergent as a `RationalReal`. Non-finite inputs
/// (`NaN`, `±Inf`) panic.
pub fn tighten_scalar(x: f64, q_max: u64) -> RationalReal {
    if !x.is_finite() {
        panic!("tighten_scalar: non-finite input ({x})");
    }
    if x == 0.0 {
        return RationalReal::from_pair(BigInt::zero(), BigInt::from(1));
    }
    let neg = x.is_sign_negative();
    let xa = x.abs();

    // Convergents h_{k}/k_{k} via the standard recurrence:
    //   a_0 = floor(x);  x_{i+1} = 1 / (x_i - a_i)
    //   h_{-1}=1, h_{-2}=0;  k_{-1}=0, k_{-2}=1
    //   h_i = a_i * h_{i-1} + h_{i-2}
    //   k_i = a_i * k_{i-1} + k_{i-2}
    // Stop once k_i > q_max; return the previous convergent.
    let mut x_curr = xa;
    let (mut h_prev2, mut h_prev1): (u128, u128) = (0, 1);
    let (mut k_prev2, mut k_prev1): (u128, u128) = (1, 0);

    // Safety bound: at most 64 levels of continued fraction for any
    // f64; we cap at 100 as defense-in-depth.
    for _ in 0..100 {
        let a = x_curr.floor() as u64;
        let h_new = a as u128 * h_prev1 + h_prev2;
        let k_new = a as u128 * k_prev1 + k_prev2;
        if k_new > q_max as u128 {
            break;
        }
        h_prev2 = h_prev1;
        h_prev1 = h_new;
        k_prev2 = k_prev1;
        k_prev1 = k_new;
        let frac = x_curr - x_curr.floor();
        if frac < 1e-30 {
            break; // exact match (or close enough)
        }
        x_curr = 1.0 / frac;
    }

    let numer_u = h_prev1;
    let denom_u = k_prev1.max(1);
    let mut numer = BigInt::from(numer_u);
    if neg {
        numer = -numer;
    }
    let denom = BigInt::from(denom_u);
    RationalReal::from_bigrational(BigRational::new(numer, denom))
}

/// Element-wise [`tighten_scalar`] over an `f64` slice. Returns a
/// `Vec<RationalReal>` in the destination thread's arena. Non-finite
/// entries panic.
pub fn tighten_vec(x: &[f64], q_max: u64) -> Vec<RationalReal> {
    x.iter().map(|&xi| tighten_scalar(xi, q_max)).collect()
}

/// Element-wise tightening over a 2D matrix block, returning a
/// `Vec<Vec<RationalReal>>`. Useful for symmetric PSD blocks.
/// Non-finite entries panic.
pub fn tighten_psd_block(block: &[Vec<f64>], q_max: u64) -> Vec<Vec<RationalReal>> {
    block.iter().map(|row| tighten_vec(row, q_max)).collect()
}

#[cfg(test)]
mod tests {
    use super::super::reset_arena;
    use super::*;

    #[test]
    fn tighten_zero_is_zero() {
        reset_arena();
        let r = tighten_scalar(0.0, 1000);
        let (n, d) = r.into_pair();
        assert_eq!(n, BigInt::zero());
        assert_eq!(d, BigInt::from(1));
        reset_arena();
    }

    #[test]
    fn tighten_recovers_exact_simple_fractions() {
        reset_arena();
        // 0.5 has convergents [1/2]; with q_max ≥ 2 we get exactly 1/2.
        let r = tighten_scalar(0.5, 100);
        assert_eq!(r.into_pair(), (BigInt::from(1), BigInt::from(2)));

        // 0.25 -> 1/4
        let r = tighten_scalar(0.25, 100);
        assert_eq!(r.into_pair(), (BigInt::from(1), BigInt::from(4)));

        // 0.75 -> 3/4
        let r = tighten_scalar(0.75, 100);
        assert_eq!(r.into_pair(), (BigInt::from(3), BigInt::from(4)));

        // 1/3 (rounded into f64) -> still 1/3 with q_max ≥ 3
        let r = tighten_scalar(1.0_f64 / 3.0, 100);
        assert_eq!(r.into_pair(), (BigInt::from(1), BigInt::from(3)));
        reset_arena();
    }

    #[test]
    fn tighten_handles_negative_values() {
        reset_arena();
        let r = tighten_scalar(-0.5, 100);
        assert_eq!(r.into_pair(), (BigInt::from(-1), BigInt::from(2)));

        let r = tighten_scalar(-2.0_f64 / 3.0, 100);
        assert_eq!(r.into_pair(), (BigInt::from(-2), BigInt::from(3)));
        reset_arena();
    }

    #[test]
    fn tighten_pi_with_growing_denominator_bounds() {
        reset_arena();
        // Famous PI convergents: 3, 22/7, 333/106, 355/113, ...
        let pi = std::f64::consts::PI;

        let r3 = tighten_scalar(pi, 5).into_pair();
        assert_eq!(r3, (BigInt::from(3), BigInt::from(1)));

        let r22 = tighten_scalar(pi, 50).into_pair();
        assert_eq!(r22, (BigInt::from(22), BigInt::from(7)));

        let r355 = tighten_scalar(pi, 200).into_pair();
        assert_eq!(r355, (BigInt::from(355), BigInt::from(113)));
        reset_arena();
    }

    #[test]
    fn tighten_vec_element_wise() {
        reset_arena();
        let xs = [0.5_f64, 0.25, -0.75, 0.0];
        let rs = tighten_vec(&xs, 100);
        assert_eq!(rs.len(), 4);
        assert_eq!(rs[0].into_pair(), (BigInt::from(1), BigInt::from(2)));
        assert_eq!(rs[1].into_pair(), (BigInt::from(1), BigInt::from(4)));
        assert_eq!(rs[2].into_pair(), (BigInt::from(-3), BigInt::from(4)));
        assert_eq!(rs[3].into_pair(), (BigInt::zero(), BigInt::from(1)));
        reset_arena();
    }

    #[test]
    fn tighten_clamps_at_denominator_bound() {
        reset_arena();
        // 0.123456789 — with very small q_max we should get a coarse
        // rational, not the exact f64 representation.
        let r = tighten_scalar(0.123_456_789, 10);
        let (_, d) = r.into_pair();
        assert!(d <= BigInt::from(10), "denom = {d} should be ≤ 10");
        reset_arena();
    }
}
