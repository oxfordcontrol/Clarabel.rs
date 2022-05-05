#![allow(non_snake_case)]
use clarabel_algebra::*;

fn test_matrix_4x4_triu() -> CscMatrix<f64>{

    // A =
    //[ 4.0  -3.0   7.0    ⋅ ]
    //[  ⋅    8.0  -1.0    ⋅ ]
    //[  ⋅     ⋅    2.0  -3.0]
    //[  ⋅     ⋅     ⋅    1.0]
    let Ap = vec![0, 1, 3, 6, 8];
    let Ai = vec![0, 0, 1, 0, 1, 2, 2, 3];
    let Ax = vec![4., -3., 8., 7., -1., 2., -3., 1.];
    CscMatrix{
        m : 4,
        n : 4,
        colptr : Ap,
        rowval : Ai,
        nzval : Ax
    }
}

fn test_matrix_3x4() -> CscMatrix<f64>{

    // A =
    //[-1.0  -17.0  6.0  10.0]
    //[ 3.0     ⋅   7.0    ⋅ ]
    //[  ⋅    -4.0   ⋅   -5.0]
    let Ap = vec![0, 2, 4, 6, 8];
    let Ai = vec![0, 1, 0, 2, 0, 1, 0, 2];
    let Ax = vec![-1.,3.,-17.,-4.,6.,7.,10.,-5.];
    CscMatrix{
        m : 3,
        n : 4,
        colptr : Ap,
        rowval : Ai,
        nzval : Ax
    }
}

#[test]
fn test_nrows_ncols_nnz_is_square() {
    let A = test_matrix_3x4();
    let B = test_matrix_4x4_triu();
    assert_eq!(A.nrows(),3);
    assert_eq!(A.ncols(),4);
    assert_eq!(B.nrows(),4);
    assert_eq!(B.ncols(),4);
    assert!(!A.is_square());
    assert!(B.is_square());
    assert_eq!(A.nnz(),8);
    assert_eq!(B.nnz(),8);
}

#[test]
fn test_col_norms() {

    let A = test_matrix_3x4();
    let mut v = vec![0.,-30.,12., 4.];  //big values should be ignored
    A.col_norms(&mut v);
    assert_eq!(v,vec![3.,17.,7.,10.]);

    let mut v = vec![0.,-30.,12., 4.];  //big values should NOT be ignored
    A.col_norms_no_reset(&mut v);
    assert_eq!(v,vec![3.,17.,12.,10.]);

}

#[test]
fn test_col_norms_sym() {

    let A = test_matrix_4x4_triu();
    let mut v = vec![0.,-30.,20., 4.];  //big values should be ignored
    A.col_norms_sym(&mut v);
    assert_eq!(v,vec![7.,8.,7.,3.]);

    let mut v = vec![0.,-30.,12., 4.];  //big values should NOT be ignored
    A.col_norms_sym_no_reset(&mut v);
    assert_eq!(v,vec![7.,8.,12.,4.]);

}

#[test]
fn test_row_norms() {

    let A = test_matrix_3x4();
    let mut v = vec![0.,-30.,12.];  //big values should be ignored
    A.row_norms(&mut v);
    assert_eq!(v,vec![17.,7.,5.]);

    let mut v = vec![0.,-30.,12.];  //big values should NOT be ignored
    A.row_norms_no_reset(&mut v);
    assert_eq!(v,vec![17.,7.,12.]);
}

#[test]
fn test_scale() {

    let mut A = test_matrix_4x4_triu();
    A.scale(2.);
    let v = vec![8., -6., 16., 14., -2., 4., -6., 2.];
    assert_eq!(A.nzval,v);
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
    assert_eq!(A.nzval,vec![-2.,-3.,-34.,-12.,12.,-7.,20.,-15.]);

    //rscale only
    let mut A = test_matrix_3x4();
    A.rscale(&r);
    assert_eq!(A.nzval,vec![-2.,6.,17.,4.,6.,7.,40.,-20.]);

    //both
    let mut A = test_matrix_3x4();
    A.lrscale(&l,&r);
    assert_eq!(A.nzval,vec![-4.,-6.,34.,12.,12.,-7.,80.,-60.]);
}

#[test]
fn test_gemv() {

    let A = test_matrix_3x4();
    let x = vec![1., -2., 3., -4.];
    let mut y = vec![5., -6., 7.];
    let a = 2.;
    let b = -3.;

    A.gemv(&mut y, MatrixShape::N, &x, a, b);
    assert_eq!(y, vec![7.,66.,35.]);

    let mut y = vec![1., -2., 3., -4.];
    let x = vec![5., -6., 7.];

    A.gemv(&mut y, MatrixShape::T, &x, a, b);
    assert_eq!(y,vec![-49.,-220.,-33.,42.]);

}

#[test]
fn test_symv() {

    let A = test_matrix_4x4_triu();
    let x = vec![1.,2.,-3.,-4.];
    let mut y = vec![0.,1.,-1., 2.];
    let a = -2.;
    let b = 3.;

    A.symv(&mut y,MatrixTriangle::Triu, &x,a,b);

    assert_eq!(y,vec![46.0,-29.0,-25.0,-4.0]);

}

#[test]
fn test_symdot() {

    let A = test_matrix_4x4_triu();
    let x = vec![1.,2.,-3.,-4.];
    let y = vec![0.,1.,-1., 2.];

    let val = A.symdot(&y,&x);

    assert_eq!(val,15.);

}
