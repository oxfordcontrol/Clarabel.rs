//! Unit tests for the MpfrFloat backend.

#![allow(unused_imports)]

use super::*;
use crate::algebra::transcendental::{RealConst, RealSentinel, Transcendental};
use num_traits::{FromPrimitive, One, Signed, Zero};

#[test]
fn mpfr_arithmetic_at_default_precision() {
    let a = MpfrFloat::from_f64(0.1).unwrap();
    let b = MpfrFloat::from_f64(0.2).unwrap();
    let s = a + b;
    let s_f = s.to_f64();
    assert!((s_f - 0.3).abs() < 1e-15, "0.1 + 0.2 ≈ 0.3, got {s_f}");
}

#[test]
fn mpfr_default_precision_is_167() {
    set_default_precision(167);
    let z = MpfrFloat::zero();
    assert_eq!(z.prec(), 167);
}

#[test]
fn mpfr_with_precision_scope_guard_restores() {
    set_default_precision(167);
    with_mpfr_precision(300, || {
        let z = MpfrFloat::zero();
        assert_eq!(z.prec(), 300);
    });
    let z = MpfrFloat::zero();
    assert_eq!(z.prec(), 167);
}

#[test]
fn mpfr_sqrt_two_squared_close_to_two() {
    set_default_precision(200);
    let two = MpfrFloat::from_i64(2).unwrap();
    let s = two.clone().sqrt();
    let back = s.clone() * s;
    let diff = (back - two).abs();
    let tol = MpfrFloat::from_f64(1e-50).unwrap();
    assert!(diff < tol, "sqrt(2)² should match 2 within 2^-50ish at 200 bits");
}

#[test]
fn mpfr_exp_ln_round_trip() {
    set_default_precision(200);
    for v in [0.5_f64, 1.0_f64, 2.0_f64, 100.0_f64] {
        let x = MpfrFloat::from_f64(v).unwrap();
        let back = x.clone().ln().exp();
        let err = (back - x).abs();
        let tol = MpfrFloat::from_f64(1e-40).unwrap();
        assert!(err < tol, "exp(ln({v})) round-trip");
    }
}

#[test]
fn mpfr_recip_of_zero_is_infinity() {
    let z = MpfrFloat::zero();
    let r = z.recip();
    // MPFR division-by-zero gives +inf with the standard rounding mode.
    assert!(<MpfrFloat as RealSentinel>::is_infinite(r));
}

#[test]
fn mpfr_pi_close_to_f64_pi() {
    set_default_precision(167);
    let pi = <MpfrFloat as RealConst>::PI();
    let pi_f = pi.to_f64();
    assert!((pi_f - std::f64::consts::PI).abs() < 1e-15);
}

#[test]
fn mpfr_is_floatt() {
    fn assert_floatt<T: crate::algebra::FloatT>() {}
    assert_floatt::<MpfrFloat>();
    let one = MpfrFloat::one();
    assert!(<MpfrFloat as RealSentinel>::is_finite(one));
}

#[cfg(feature = "serde")]
#[test]
fn mpfr_serde_round_trip() {
    set_default_precision(200);
    let r = MpfrFloat::from_f64(0.1).unwrap();
    let s = serde_json::to_string(&r).unwrap();
    let back: MpfrFloat = serde_json::from_str(&s).unwrap();
    let diff = (back - r).abs();
    let tol = MpfrFloat::from_f64(1e-50).unwrap();
    assert!(diff < tol, "serde round-trip preserves value");
}

// ---------------------------------------------------------------------------
// Dense LAPACK backend tests (`native_lapack.rs`).
//
// Added 2026-08-11. Before this, all 9 tests in this file covered *scalar*
// arithmetic only; the 329 lines of hand-rolled dense linear algebra —
// including `xsyevr`, on which PSD-cone projection depends — had **zero**
// coverage. A wrong eigenvalue in a cone projection does not crash; it
// returns a plausible wrong answer, so this is the highest-risk untested
// surface in the backend.
//
// LAPACK convention throughout: column-major, `a[j*lda + i]` is A[i][j].
// ---------------------------------------------------------------------------

// native_lapack is only compiled with `sdp` (that is where the X*Scalar
// traits are declared), so these tests follow it.
#[cfg(all(test, feature = "sdp"))]
mod lapack_tests {
    use super::*;
    // The X*Scalar traits must be in scope to call their methods on MpfrFloat.
    use crate::algebra::{XpotrfScalar, XsyevrScalar};

    fn f(x: f64) -> MpfrFloat {
        MpfrFloat::from_f64(x).unwrap()
    }

    /// Run `xsyevr` on a symmetric `n×n` given column-major, returning
    /// (eigenvalues ascending, eigenvectors column-major).
    fn syevr(n: usize, a_in: &[MpfrFloat], uplo: u8, jobz: u8)
        -> (Vec<MpfrFloat>, Vec<MpfrFloat>)
    {
        let mut a = a_in.to_vec();
        let mut w = vec![MpfrFloat::zero(); n];
        let mut z = vec![MpfrFloat::zero(); n * n];
        let mut m = 0i32;
        let mut info = 0i32;
        let mut isuppz = vec![0i32; 2 * n];
        let mut work = vec![MpfrFloat::zero(); 1];
        let mut iwork = vec![0i32; 1];
        MpfrFloat::xsyevr(
            jobz, b'A', uplo, n as i32, &mut a, n as i32,
            MpfrFloat::zero(), MpfrFloat::zero(), 0, 0, MpfrFloat::zero(),
            &mut m, &mut w, &mut z, n as i32, &mut isuppz,
            &mut work, 1, &mut iwork, 1, &mut info,
        );
        assert_eq!(info, 0, "xsyevr reported info={info}");
        assert_eq!(m, n as i32, "xsyevr returned m={m}, expected {n}");
        (w, z)
    }

    fn close(a: &MpfrFloat, b: f64, tol: f64, what: &str) {
        let d = (a.clone() - MpfrFloat::from_f64(b).unwrap()).abs().to_f64();
        assert!(d < tol, "{what}: got {}, want {b} (|diff| = {d:e})", a.to_f64());
    }

    #[test]
    fn mpfr_syevr_diagonal_spectrum_ascending() {
        // A = diag(3, 1, 2): eigenvalues are the diagonal, returned ascending.
        let mut a = vec![MpfrFloat::zero(); 9];
        a[0] = f(3.0);
        a[4] = f(1.0);
        a[8] = f(2.0);
        let (w, _) = syevr(3, &a, b'L', b'V');
        close(&w[0], 1.0, 1e-40, "w0");
        close(&w[1], 2.0, 1e-40, "w1");
        close(&w[2], 3.0, 1e-40, "w2");
    }

    #[test]
    fn mpfr_syevr_2x2_closed_form() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3 exactly.
        let a = vec![f(2.0), f(1.0), f(1.0), f(2.0)];
        let (w, _) = syevr(2, &a, b'L', b'V');
        close(&w[0], 1.0, 1e-40, "w0");
        close(&w[1], 3.0, 1e-40, "w1");
    }

    #[test]
    fn mpfr_syevr_eigenvectors_orthonormal() {
        // Zᵀ Z = I to high precision.
        let a = vec![f(4.0), f(1.0), f(2.0), f(1.0), f(3.0), f(0.5), f(2.0), f(0.5), f(5.0)];
        let (_, z) = syevr(3, &a, b'L', b'V');
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = MpfrFloat::zero();
                for k in 0..3 {
                    dot = dot + z[i * 3 + k].clone() * z[j * 3 + k].clone();
                }
                let want = if i == j { 1.0 } else { 0.0 };
                close(&dot, want, 1e-38, &format!("ZᵀZ[{i}][{j}]"));
            }
        }
    }

    #[test]
    fn mpfr_syevr_reconstructs_the_matrix() {
        // A = Z Λ Zᵀ — checks eigenvalues AND eigenvectors together.
        let a = vec![f(4.0), f(1.0), f(2.0), f(1.0), f(3.0), f(0.5), f(2.0), f(0.5), f(5.0)];
        let (w, z) = syevr(3, &a, b'L', b'V');
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = MpfrFloat::zero();
                for k in 0..3 {
                    acc = acc + z[k * 3 + i].clone() * w[k].clone() * z[k * 3 + j].clone();
                }
                close(&acc, a[j * 3 + i].to_f64(), 1e-36, &format!("A[{i}][{j}]"));
            }
        }
    }

    #[test]
    fn mpfr_syevr_uplo_upper_agrees_with_lower() {
        // The impl branches on `uplo`; both readings must give one spectrum.
        // Store only the referenced triangle, garbage elsewhere, to prove the
        // untouched triangle really is ignored.
        let lower = vec![f(4.0), f(1.0), f(2.0), f(99.0), f(3.0), f(0.5), f(99.0), f(99.0), f(5.0)];
        let upper = vec![f(4.0), f(99.0), f(99.0), f(1.0), f(3.0), f(99.0), f(2.0), f(0.5), f(5.0)];
        let (wl, _) = syevr(3, &lower, b'L', b'N');
        let (wu, _) = syevr(3, &upper, b'U', b'N');
        for k in 0..3 {
            let d = (wl[k].clone() - wu[k].clone()).abs().to_f64();
            assert!(d < 1e-38, "uplo mismatch at {k}: L={} U={}", wl[k].to_f64(), wu[k].to_f64());
        }
    }

    #[test]
    fn mpfr_syevr_jobz_n_matches_jobz_v_eigenvalues() {
        let a = vec![f(4.0), f(1.0), f(2.0), f(1.0), f(3.0), f(0.5), f(2.0), f(0.5), f(5.0)];
        let (wv, _) = syevr(3, &a, b'L', b'V');
        let (wn, _) = syevr(3, &a, b'L', b'N');
        for k in 0..3 {
            let d = (wv[k].clone() - wn[k].clone()).abs().to_f64();
            assert!(d < 1e-38, "jobz mismatch at {k}");
        }
    }

    /// **The test f64 cannot pass** — the reason this backend exists.
    ///
    /// `A = [[1+δ/2, −δ/2], [−δ/2, 1+δ/2]]` has eigenvalues exactly `1` and
    /// `1+δ`. With `δ = 2⁻¹⁰⁰`, the entry `1 + δ/2 = 1 + 2⁻¹⁰¹` needs 102
    /// mantissa bits, so **f64 cannot even store A** — it rounds to `1.0` and
    /// the eigenvalue split vanishes. At the 167-bit default the split must
    /// survive and be recovered to relative accuracy.
    #[test]
    fn mpfr_syevr_resolves_a_gap_f64_cannot_represent() {
        set_default_precision(167);

        // f64 provably loses the matrix entry:
        assert_eq!(1.0f64 + 2f64.powi(-101), 1.0f64,
                   "premise: 1 + 2^-101 is not representable in f64");

        let delta = f(2f64.powi(-100));
        let half_delta = f(2f64.powi(-101));
        let diag = MpfrFloat::one() + half_delta.clone();
        let off = MpfrFloat::zero() - half_delta;
        let a = vec![diag.clone(), off.clone(), off, diag];

        let (w, _) = syevr(2, &a, b'L', b'N');

        // Smaller eigenvalue is exactly 1.
        let e0 = (w[0].clone() - MpfrFloat::one()).abs();
        assert!(e0.to_f64() < 1e-40, "w0 should be 1, got {}", w[0].to_f64());

        // The gap is δ, recovered to better than 1e-6 relative.
        let gap = w[1].clone() - w[0].clone();
        let rel = ((gap - delta.clone()) / delta.clone()).abs().to_f64();
        assert!(rel < 1e-6,
                "gap should be 2^-100 to 1e-6 relative; relative error {rel:e}");
    }

    #[test]
    fn mpfr_potrf_cholesky_reconstructs_lower() {
        // A = L Lᵀ for a known SPD matrix.
        let a0 = vec![f(4.0), f(2.0), f(-2.0), f(2.0), f(10.0), f(2.0), f(-2.0), f(2.0), f(5.0)];
        let mut a = a0.clone();
        let mut info = 0i32;
        MpfrFloat::xpotrf(b'L', 3, &mut a, 3, &mut info);
        assert_eq!(info, 0, "xpotrf reported info={info}");
        for i in 0..3usize {
            for j in 0..=i {
                let mut acc = MpfrFloat::zero();
                for k in 0..=j {
                    acc = acc + a[k * 3 + i].clone() * a[k * 3 + j].clone();
                }
                close(&acc, a0[j * 3 + i].to_f64(), 1e-38, &format!("LLᵀ[{i}][{j}]"));
            }
        }
    }
}
