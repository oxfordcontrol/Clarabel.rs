use super::*;
use crate::algebra::{CscMatrix, FloatT};
extern crate amd;

#[cfg(test)]

fn test_matrix_4x4() -> CscMatrix<f64> {
    // A =
    //[ 8.0  -3.0   2.0    ⋅ ]
    //[  ⋅    8.0  -1.0    ⋅ ]
    //[  ⋅     ⋅    8.0  -1.0]
    //[  ⋅     ⋅     ⋅    1.0]
    let Ap = vec![0, 1, 3, 6, 8];
    let Ai = vec![0, 0, 1, 0, 1, 2, 2, 3];
    let Ax = vec![8., -3., 8., 2., -1., 8., -1., 1.];
    CscMatrix {
        m: 4,
        n: 4,
        colptr: Ap,
        rowval: Ai,
        nzval: Ax,
    }
}

fn inf_norm_diff<T: FloatT>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b)
        .fold(T::zero(), |acc, (x, y)| T::max(acc, T::abs(*x - *y)))
}

// tests some of the private functions of QDLDL.  Configured
// as submodule from lib.rs to expose internals.

#[test]
fn test_invperm() {
    let perm = vec![3, 0, 2, 1];
    let iperm = _invperm(&perm);
    assert_eq!(iperm, vec![1, 3, 2, 0]);
}

//test fail on bad permutation
#[test]
#[should_panic]
fn test_invperm_bad_perm_panic1() {
    let perm = vec![3, 0, 2, 0]; //repeated index
    _invperm(&perm);
}

#[test]
#[should_panic]
fn test_invperm_bad_perm_panic2() {
    let perm = vec![4, 0, 2, 1]; //index too big
    _invperm(&perm);
}

#[test]
fn test_permute() {
    let perm = vec![3, 0, 2, 1]; //index too big
    let b = vec![1., 2., 3., 4.];
    let mut x = vec![0.; 4];
    let mut y = vec![0.; 4];

    _permute(&mut x, &b, &perm);
    assert_eq!(x, vec![4., 1., 3., 2.]);

    _ipermute(&mut y, &x, &perm);
    assert_eq!(y, b);
}

#[test]
fn test_solve_from_factors() {
    //L =
    //[ ⋅    ⋅     ⋅    ⋅ ]
    //[1.0   ⋅     ⋅    ⋅ ]
    //[2.0  1.0    ⋅    ⋅ ]
    //[ ⋅   7.0  -3.0   ⋅ ]

    let Lp = vec![0, 2, 4, 5, 5];
    let Li = vec![1, 2, 2, 3, 3];
    let Lx = vec![1., 2., 1., 7., -3.];
    let _d = vec![4., -1., -2., 1.];
    let dinv = [0.25, -1.0, -0.5, 1.0];
    let x = vec![-3., 2., 1., 4.];

    //(I+L)x = b.  Back solve on b in place.
    let mut b = vec![-3., -1., -3., 15.];
    _lsolve(&Lp, &Li, &Lx, &mut b);
    assert_eq!(b, x);

    //(I+L')x = b.  Back solve on b in place.
    let mut b = vec![1., 31., -11., 4.];
    _ltsolve(&Lp, &Li, &Lx, &mut b);
    assert_eq!(b, x);

    //(I+L)*D*(I+L)*x = b.  Back solve on b in place;
    let mut b = vec![4., -27., -1., -279.];
    _solve(&Lp, &Li, &Lx, &dinv, &mut b);
    assert_eq!(b, x);
}

#[test]
fn test_etree() {
    let n = 4;
    let A = test_matrix_4x4();
    let mut Lnz = vec![0; n];
    let mut iwork = vec![0; 3 * n];
    let mut etree = vec![0; n];

    let _out = _etree(
        A.nrows(),
        &A.colptr,
        &A.rowval,
        &mut iwork,
        &mut Lnz,
        &mut etree,
    )
    .unwrap();

    assert_eq!(etree, vec![1, 2, 3, QDLDL_UNKNOWN]);
}

#[test]
fn test_amd() {
    let A = test_matrix_4x4();
    let (perm, iperm) = _get_amd_ordering(&A);
    assert_eq!(perm, [3, 0, 1, 2]);
    assert_eq!(iperm, [1, 2, 3, 0]);
}

#[test]
fn test_permute_symmetric() {
    //no permutation at all
    let A = test_matrix_4x4();
    let iperm: Vec<usize> = vec![0, 1, 2, 3];
    let (P, AtoPAPt) = _permute_symmetric(&A, &iperm);

    assert_eq!(&A.colptr, &P.colptr);
    assert_eq!(&A.rowval, &P.rowval);
    assert_eq!(&A.nzval, &P.nzval);
    let linearidx: Vec<usize> = (0..AtoPAPt.len()).collect();
    assert_eq!(&linearidx, &AtoPAPt);

    //test with a permutation.  NB: the permutation
    //implemented in QDLDL produces a result in which entries
    //are not ordering by increasing row number within
    //each column, so caution is required when comparing
    //w.r.t. other tools (i.e. Matlab/Julia/Python etc)

    let mut A = test_matrix_4x4();

    //set the problem data to increasing values columnwise
    for i in 0..A.nzval.len() {
        A.nzval[i] = i as f64 + 1.;
    }

    let perm: Vec<usize> = vec![2, 3, 0, 1];
    let iperm = _invperm(&perm);
    let (P, _) = _permute_symmetric(&A, &iperm);

    assert_eq!(&P.colptr, &vec![0, 1, 3, 5, 8]);
    assert_eq!(&P.rowval, &vec![0, 0, 1, 2, 0, 2, 3, 0]);
    assert_eq!(&P.nzval, &vec![6.0, 7.0, 8.0, 1.0, 4.0, 2.0, 3.0, 5.0]);
}

#[test]
fn test_settings_builder() {
    //NB: the default regularize_eps is 1e-12.  Use this
    //as a reference point throughout
    let expected_regularize_eps = 1e-12;

    //check that defaults appear when not using builder
    let opts = QDLDLSettings::<f64>::default();
    assert_eq!(opts.regularize_eps, expected_regularize_eps);

    //same thing through the builder
    let opts = QDLDLSettingsBuilder::<f64>::default().build().unwrap();
    assert_eq!(opts.regularize_eps, expected_regularize_eps);

    //and now a custom builder
    let opts = QDLDLSettingsBuilder::default()
        .perm(vec![0, 1, 2, 3])
        .logical(false)
        .regularize_enable(true)
        .regularize_eps(1e-3)
        .regularize_delta(1e-3)
        .build()
        .unwrap();

    assert_eq!(opts.regularize_eps, 1e-3);
    assert_eq!(opts.regularize_delta, 1e-3);
}

#[test]
fn test_solve_basic() {
    let A = test_matrix_4x4();

    //default settings but no permutation
    let opts = QDLDLSettingsBuilder::default()
        .perm(vec![0, 1, 2, 3])
        .build()
        .unwrap();

    let mut factors = QDLDLFactorisation::new(&A, Some(opts));
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);

    //now with all defaults, including amd
    let mut factors = QDLDLFactorisation::new(&A, None);
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);

    //user specified permutation
    let opts = QDLDLSettingsBuilder::<f64>::default()
        .perm(vec![3, 0, 2, 1])
        .build()
        .unwrap();
    let mut factors = QDLDLFactorisation::new(&A, Some(opts));
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);
}

#[test]
#[should_panic]
fn test_solve_logical() {
    let A = test_matrix_4x4();
    //logical first, then refactor and solve
    let opts = QDLDLSettingsBuilder::default()
        .logical(true)
        .build()
        .unwrap();

    let mut factors = QDLDLFactorisation::new(&A, Some(opts));
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.solve(&mut b);
}

#[test]
fn test_solve_logical_refactor() {
    let A = test_matrix_4x4();
    //logical first, then refactor and solve
    let opts = QDLDLSettingsBuilder::default()
        .logical(true)
        .build()
        .unwrap();

    let mut factors = QDLDLFactorisation::new(&A, Some(opts));
    let x = [1., -2., 3., -4.];
    let mut b = [20.0, -22.0, 32.0, -7.0];
    //solves in place
    factors.refactor();
    factors.solve(&mut b);
    assert!(inf_norm_diff(&x, &b) <= 1e-8);
}
