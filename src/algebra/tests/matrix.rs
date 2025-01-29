#![allow(non_snake_case)]
use crate::algebra::*;

fn test_matrix_4x4_triu() -> CscMatrix<f64> {
    // A =
    //[ 4.0  -3.0   7.0    ⋅ ]
    //[  ⋅    8.0  -1.0    ⋅ ]
    //[  ⋅     ⋅    2.0  -3.0]
    //[  ⋅     ⋅     ⋅    1.0]
    let Ap = vec![0, 1, 3, 6, 8];
    let Ai = vec![0, 0, 1, 0, 1, 2, 2, 3];
    let Ax = vec![4., -3., 8., 7., -1., 2., -3., 1.];
    CscMatrix::new(4, 4, Ap, Ai, Ax)
}

fn test_matrix_4x4() -> CscMatrix<f64> {
    // A =
    //[ 4.0  -3.0   7.0    ⋅ ]
    //[  ⋅    8.0  -1.0    ⋅ ]
    //[ 1.0    ⋅    2.0  -3.0]
    //[  ⋅   -1.0    ⋅    1.0]

    //NB: same as above, but with tril entries
    let Ap = vec![0, 2, 5, 8, 10];
    let Ai = vec![0, 2, 0, 1, 3, 0, 1, 2, 2, 3];
    let Ax = vec![4., 1., -3., 8., -1., 7., -1., 2., -3., 1.];
    CscMatrix::new(4, 4, Ap, Ai, Ax)
}

fn test_matrix_4x4_triu_2() -> CscMatrix<f64> {
    // A =
    //[ 4.0  -3.0   7.0    ⋅ ]
    //[  ⋅     ⋅   -1.0    ⋅ ]
    //[  ⋅     ⋅    2.0  -3.0]
    //[  ⋅     ⋅     ⋅    1.0]

    // NB: same as 4x4_triu, but with missing diagonal entry

    let Ap = vec![0, 1, 2, 5, 7];
    let Ai = vec![0, 0, 0, 1, 2, 2, 3];
    let Ax = vec![4., -3., 7., -1., 2., -3., 1.];
    CscMatrix::new(4, 4, Ap, Ai, Ax)
}

fn test_matrix_4x4_2() -> CscMatrix<f64> {
    // A =
    //[ 4.0  -3.0   7.0    ⋅ ]
    //[  ⋅    8.0  -1.0    ⋅ ]
    //[ 1.0    ⋅    2.0  -3.0]
    //[  ⋅   -1.0    ⋅    1.0]

    //NB: same as above, but with tril entries
    let Ap = vec![0, 2, 4, 7, 9];
    let Ai = vec![0, 2, 0, 3, 0, 1, 2, 2, 3];
    let Ax = vec![4., 1., -3., -1., 7., -1., 2., -3., 1.];
    CscMatrix::new(4, 4, Ap, Ai, Ax)
}

fn test_matrix_3x4() -> CscMatrix<f64> {
    // A =
    //[-1.0  -17.0  6.0  10.0]
    //[ 3.0     ⋅   7.0    ⋅ ]
    //[  ⋅    -4.0   ⋅   -5.0]
    let Ap = vec![0, 2, 4, 6, 8];
    let Ai = vec![0, 1, 0, 2, 0, 1, 0, 2];
    let Ax = vec![-1., 3., -17., -4., 6., 7., 10., -5.];
    CscMatrix::new(3, 4, Ap, Ai, Ax)
}

#[test]
fn test_nrows_ncols_nnz_is_square() {
    let A = test_matrix_3x4();
    let B = test_matrix_4x4_triu();
    assert_eq!(A.nrows(), 3);
    assert_eq!(A.ncols(), 4);
    assert_eq!(B.nrows(), 4);
    assert_eq!(B.ncols(), 4);
    assert!(!A.is_square());
    assert!(B.is_square());
    assert_eq!(A.nnz(), 8);
    assert_eq!(B.nnz(), 8);
}

#[test]
fn test_check_format() {
    assert!(test_matrix_3x4().check_format().is_ok());
    assert!(test_matrix_4x4().check_format().is_ok());
    assert!(test_matrix_4x4_triu().check_format().is_ok());
    assert!(test_matrix_4x4_2().check_format().is_ok());
    assert!(test_matrix_4x4_triu_2().check_format().is_ok());

    //bad col dimension
    let mut A = test_matrix_4x4();
    A.n = 10;
    assert!(A.check_format().is_err());

    //rowval / numeric value length mismatch (rowval too short)
    let mut A = test_matrix_4x4();
    A.rowval.pop();
    assert!(A.check_format().is_err());

    //rowval / numeric value length mismatch (nzval too short)
    let mut A = test_matrix_4x4();
    A.nzval.pop();
    assert!(A.check_format().is_err());

    //bad col ptr (end value wrong)
    let mut A = test_matrix_4x4();
    *A.colptr.last_mut().unwrap() = 100;
    assert!(A.check_format().is_err());

    //bad col ptr (first value wrong)
    let mut A = test_matrix_4x4();
    A.colptr[0] = 100;
    assert!(A.check_format().is_err());

    //bad col ptr (empty)
    let mut A = test_matrix_4x4();
    A.colptr = vec![];
    assert!(A.check_format().is_err());

    //badly ordered rows
    let mut A = test_matrix_4x4();
    //was Ai = vec![0, 2, 0, 1, 3, 0, 1, 2, 2, 3]
    A.rowval = vec![0, 2, 0, 3, 1, 0, 1, 2, 2, 3];
    assert!(A.check_format().is_err());

    //repeated matrix entry
    let mut A = test_matrix_4x4();
    //was Ai = vec![0, 2, 0, 1, 3, 0, 1, 2, 2, 3]
    A.rowval = vec![0, 2, 0, 1, 1, 0, 1, 2, 2, 3];
    assert!(A.check_format().is_err());

    //row index out of bounds
    let mut A = test_matrix_4x4();
    //was Ai = vec![0, 2, 0, 1, 3, 0, 1, 2, 2, 3]
    A.rowval = vec![0, 2, 0, 1, 4, 0, 1, 2, 2, 3];
    assert!(A.check_format().is_err());
}

#[test]
fn test_row_sums() {
    let A = test_matrix_3x4();
    let mut v = vec![0.0; 3];
    A.row_sums(&mut v);
    assert_eq!(v, vec![-2., 10., -9.]);
}

#[test]
fn test_col_sums() {
    let A = test_matrix_3x4();
    let mut v = vec![0.0; 4];
    A.col_sums(&mut v);
    assert_eq!(v, vec![2., -21., 13., 5.]);
}

#[test]
fn test_col_norms() {
    let A = test_matrix_3x4();
    let mut v = vec![0., -30., 12., 4.]; //big values should be ignored
    A.col_norms(&mut v);
    assert_eq!(v, vec![3., 17., 7., 10.]);

    let mut v = vec![0., -30., 12., 4.]; //big values should NOT be ignored
    A.col_norms_no_reset(&mut v);
    assert_eq!(v, vec![3., 17., 12., 10.]);
}

#[test]
fn test_col_norms_sym() {
    let A = test_matrix_4x4_triu();
    let mut v = vec![0., -30., 20., 4.]; //big values should be ignored
    A.col_norms_sym(&mut v);
    assert_eq!(v, vec![7., 8., 7., 3.]);

    let mut v = vec![0., -30., 12., 4.]; //big values should NOT be ignored
    A.col_norms_sym_no_reset(&mut v);
    assert_eq!(v, vec![7., 8., 12., 4.]);
}

#[test]
fn test_row_norms() {
    let A = test_matrix_3x4();
    let mut v = vec![0., -30., 12.]; //big values should be ignored
    A.row_norms(&mut v);
    assert_eq!(v, vec![17., 7., 5.]);

    let mut v = vec![0., -30., 12.]; //big values should NOT be ignored
    A.row_norms_no_reset(&mut v);
    assert_eq!(v, vec![17., 7., 12.]);
}

#[test]
fn test_scale() {
    let mut A = test_matrix_4x4_triu();
    A.scale(2.);
    let v = vec![8., -6., 16., 14., -2., 4., -6., 2.];
    assert_eq!(A.nzval, v);
}

#[test]
fn test_lrscaling() {
    // A =
    //[-1.0  -17.0  6.0  10.0]
    //[ 3.0     ⋅   7.0    ⋅ ]
    //[  ⋅    -4.0   ⋅   -5.0]
    let l = vec![2., -1., 3.];
    let r = vec![2., -1., 1., 4.];

    //lscale only
    let mut A = test_matrix_3x4();
    A.lscale(&l);
    assert_eq!(A.nzval, vec![-2., -3., -34., -12., 12., -7., 20., -15.]);

    //rscale only
    let mut A = test_matrix_3x4();
    A.rscale(&r);
    assert_eq!(A.nzval, vec![-2., 6., 17., 4., 6., 7., 40., -20.]);

    //both
    let mut A = test_matrix_3x4();
    A.lrscale(&l, &r);
    assert_eq!(A.nzval, vec![-4., -6., 34., 12., 12., -7., 80., -60.]);
}

#[test]
fn test_gemv() {
    let A = test_matrix_3x4();
    let x = vec![1., -2., 3., -4.];
    let mut y = vec![5., -6., 7.];
    let a = 2.;
    let b = -3.;

    A.gemv(&mut y, &x, a, b);
    assert_eq!(y, vec![7., 66., 35.]);

    let mut y = vec![1., -2., 3., -4.];
    let x = vec![5., -6., 7.];

    A.t().gemv(&mut y, &x, a, b);
    assert_eq!(y, vec![-49., -220., -33., 42.]);
}

#[test]
fn test_symv() {
    let A = test_matrix_4x4_triu();
    let x = vec![1., 2., -3., -4.];
    let mut y = vec![0., 1., -1., 2.];
    let a = -2.;
    let b = 3.;

    let Asym = A.sym();
    Asym.symv(&mut y, &x, a, b);

    assert_eq!(y, vec![46.0, -29.0, -25.0, -4.0]);
}

#[test]
fn test_quad_form() {
    let A = test_matrix_4x4_triu();
    let x = vec![1., 2., -3., -4.];
    let y = vec![0., 1., -1., 2.];

    let val = A.quad_form(&y, &x);

    assert_eq!(val, 15.);
}

#[test]
fn test_matrix_to_triu() {
    let Afull = test_matrix_4x4();
    let Atriu = test_matrix_4x4_triu();

    let B = Afull.to_triu();
    assert_eq!(B, Atriu);
}

#[test]
fn test_matrix_to_triu_missing_diag() {
    let Afull = test_matrix_4x4_2();
    let Atriu = test_matrix_4x4_triu_2();
    let B = Afull.to_triu();
    assert_eq!(B, Atriu);
}

#[test]
fn test_matrix_to_triu_identity() {
    let A = CscMatrix::<f64>::identity(4);
    let B = A.to_triu();
    assert_eq!(B, A);
}

#[test]
fn test_matrix_to_triu_empty() {
    let A = CscMatrix::<f64>::zeros((5, 5));
    let B = A.to_triu();
    assert_eq!(B, A);
}

#[test]
#[should_panic]
fn test_matrix_to_triu_notsquare() {
    let A = CscMatrix::<f64>::zeros((5, 4));
    let B = A.to_triu();
    assert_eq!(B, A);
}

#[test]
fn test_matrix_select_rows() {
    let A = test_matrix_4x4();

    // reduce by one row
    let rowidx = vec![true, true, false, true];
    let Ared = A.select_rows(&rowidx);

    assert_eq!(Ared.ncols(), 4);
    assert_eq!(Ared.nrows(), 3);
    assert_eq!(Ared.colptr, vec![0, 1, 4, 6, 7]);
    assert_eq!(Ared.rowval, vec![0, 0, 1, 2, 0, 1, 2]);
    assert_eq!(Ared.nzval, vec![4.0, -3.0, 8.0, -1.0, 7.0, -1.0, 1.0]);

    // reduce by three rows
    let rowidx = vec![false, false, false, true];
    let Ared = A.select_rows(&rowidx);

    assert_eq!(Ared.ncols(), 4);
    assert_eq!(Ared.nrows(), 1);
    assert_eq!(Ared.colptr, vec![0, 0, 1, 1, 2]);
    assert_eq!(Ared.rowval, vec![0, 0]);
    assert_eq!(Ared.nzval, vec![-1.0, 1.0]);

    // reduce all rows
    let rowidx = vec![false; 4];
    let Ared = A.select_rows(&rowidx);

    assert_eq!(Ared.ncols(), 4);
    assert_eq!(Ared.nrows(), 0);
    assert_eq!(Ared.colptr, vec![0, 0, 0, 0, 0]);
    assert_eq!(Ared.rowval, Vec::<usize>::new());
    assert_eq!(Ared.nzval, Vec::<f64>::new());
}
