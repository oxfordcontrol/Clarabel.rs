#![cfg(all(feature = "sdp", feature = "mpfr"))]

use clarabel::algebra::*;
use num_traits::Signed;

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

#[test]
fn test_native_lapack_eigen_degenerate() {
    let mut a = vec![to_mpfr(2.0), to_mpfr(0.0), to_mpfr(0.0), to_mpfr(2.0), to_mpfr(0.0), to_mpfr(2.0)];
    let mut w = vec![to_mpfr(0.0); 3];
    let mut z = vec![to_mpfr(0.0); 9];
    let mut isuppz = vec![0i32; 6];
    let mut work = vec![to_mpfr(0.0); 26];
    let mut iwork = vec![0i32; 10];
    let mut m = 0;
    
    // Xsyevr requires standard eigenvalue decomp API.
    <MpfrFloat as BlasFloatT>::xsyevr(
        b'V', b'A', b'U', 3, &mut a, 3, to_mpfr(0.0), to_mpfr(0.0), 0, 0, to_mpfr(1e-10), &mut m, &mut w, &mut z, 3, &mut isuppz, &mut work, 26, &mut iwork, 10, &mut 0i32
    );

    // Eigenvalues should be [2.0, 2.0, 2.0]
    for i in 0..3 {
        assert!((w[i] - to_mpfr(2.0)).abs() < to_mpfr(1e-8));
    }
}

#[test]
fn test_native_lapack_cholesky_singular() {
    let mut a = vec![to_mpfr(1.0), to_mpfr(1.0), to_mpfr(1.0), to_mpfr(1.0), to_mpfr(1.0), to_mpfr(1.0), to_mpfr(1.0), to_mpfr(1.0), to_mpfr(1.0)];
    let mut info = 0;
    <MpfrFloat as BlasFloatT>::xpotrf(b'U', 3, &mut a, 3, &mut info);
    assert!(info > 0);
}

#[test]
fn test_native_lapack_matrix_mult() {
    let a = vec![to_mpfr(1.0), to_mpfr(2.0), to_mpfr(3.0), to_mpfr(4.0)]; // 2x2 matrix
    let mut c = vec![to_mpfr(0.0), to_mpfr(0.0), to_mpfr(0.0), to_mpfr(0.0)];
    <MpfrFloat as BlasFloatT>::xsyrk(b'U', b'N', 2, 2, to_mpfr(1.0), &a, 2, to_mpfr(0.0), &mut c, 2);
    // C = A * A^T
    assert!((c[0] - to_mpfr(10.0)).abs() < to_mpfr(1e-8));
    assert!((c[2] - to_mpfr(14.0)).abs() < to_mpfr(1e-8)); // Note: column-major, upper triangle
    assert!((c[3] - to_mpfr(20.0)).abs() < to_mpfr(1e-8));
}
