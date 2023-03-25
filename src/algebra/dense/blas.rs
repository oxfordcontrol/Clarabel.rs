#![cfg_attr(rustfmt, rustfmt_skip)]
#![allow(clippy::too_many_arguments)]

extern crate openblas_src;
use lapack::*;
use blas::*;

pub trait BlasFloatT: 
    private::BlasFloatSealed 
    + XsyevrScalar 
    + XpotrfScalar 
    + XgesddScalar 
    + XgesvdScalar 
    + XgemmScalar 
    + XgemvScalar 
    + XsymvScalar 
    + XsyrkScalar
{}
impl BlasFloatT for f32 {}
impl BlasFloatT for f64 {}

mod private {
 pub trait BlasFloatSealed {}
 impl BlasFloatSealed for f32 {}
 impl BlasFloatSealed for f64 {}
}


// --------------------------------------
// ?syevr : Symmetric eigen decomposition
// --------------------------------------

pub trait XsyevrScalar: Sized {
    fn xsyevr(
        jobz: u8, range: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, vl: Self, vu: Self, il: i32, iu: i32, 
        abstol: Self, m: &mut i32, w: &mut [Self], z: &mut [Self], ldz: i32, isuppz: &mut [i32], 
        work: &mut [Self], lwork: i32, iwork: &mut [i32], liwork: i32, info: &mut i32,
    );
}

macro_rules! impl_blas_xsyevr {
    ($T:ty, $XSYEVR:path) => {
        impl XsyevrScalar for $T {
            fn xsyevr(
                jobz: u8, range: u8, uplo: u8, n: i32, a: &mut [Self], lda: i32, vl: Self, vu: Self, il: i32, iu: i32, 
                abstol: Self, m: &mut i32, w: &mut [Self], z: &mut [Self], ldz: i32, isuppz: &mut [i32], 
                work: &mut [$T], lwork: i32, iwork: &mut [i32], liwork: i32, info: &mut i32,
            ) {
                unsafe{
                    $XSYEVR(
                        jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, m, 
                        w, z, ldz, isuppz, work, lwork, iwork, liwork, info,
                    );
                }
            }
        }
    };
}

impl_blas_xsyevr!(f32, ssyevr);
impl_blas_xsyevr!(f64, dsyevr);

// --------------------------------------
// ?potrf : Cholesky decomposition
// --------------------------------------

pub trait XpotrfScalar: Sized {
    fn xpotrf(
        uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32
    );
}

macro_rules! impl_blas_xpotrf{
    ($T:ty, $XPOTRF:path) => {
        impl XpotrfScalar for $T {
            fn xpotrf(
                uplo: u8, n: i32, a: &mut [Self], lda: i32, info: &mut i32
            ) {
                unsafe{
                    $XPOTRF(
                        uplo, n, a, lda, info
                    );
                }
            }
        }
    };
}

impl_blas_xpotrf!(f32, spotrf);
impl_blas_xpotrf!(f64, dpotrf);


// --------------------------------------
// ?gesdd : SVD (divide and conquer method)
// --------------------------------------

pub trait XgesddScalar: Sized {
    fn xgesdd(
        jobz: u8, m: i32, n: i32, a: &mut [Self], lda: i32, 
        s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32, 
        work: &mut [Self], lwork: i32, iwork: &mut [i32], info: &mut i32
    );
}

macro_rules! impl_blas_xgesdd{
    ($T:ty, $XGESDD:path) => {
        impl XgesddScalar for $T {
            fn xgesdd(
                jobz: u8, m: i32, n: i32, a: &mut [Self], lda: i32, 
                s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32, 
                work: &mut [Self], lwork: i32, iwork: &mut [i32], info: &mut i32
            ) {
                unsafe{
                    $XGESDD(
                        jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, iwork, info
                    );
                }
            }
        }
    };
}

impl_blas_xgesdd!(f32, sgesdd);
impl_blas_xgesdd!(f64, dgesdd);


// --------------------------------------
// ?gesvd : SVD (QR method)
// --------------------------------------

pub trait XgesvdScalar: Sized {
    fn xgesvd(
        jobu: u8, jobvt: u8,m: i32, n: i32, a: &mut [Self], lda: i32, 
        s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32, 
        work: &mut [Self], lwork: i32, info: &mut i32
    );
}

macro_rules! impl_blas_xgesvd{
    ($T:ty, $XGESVD:path) => {
        impl XgesvdScalar for $T {
            fn xgesvd(
                jobu: u8, jobvt: u8,m: i32, n: i32, a: &mut [Self], lda: i32, 
                s: &mut [Self], u: &mut [Self], ldu: i32, vt: &mut [Self], ldvt: i32, 
                work: &mut [Self], lwork: i32, info: &mut i32
            ) {
                unsafe{
                    $XGESVD(
                        jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, info
                    );
                }
            }
        }
    };
}

impl_blas_xgesvd!(f32, sgesvd);
impl_blas_xgesvd!(f64, dgesvd);


// --------------------------------------
// ?gemm : matrix matrix multiply
// --------------------------------------

pub trait XgemmScalar: Sized {
    fn xgemm(
        transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self], 
        lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32
    );
}

macro_rules! impl_blas_gemm {
    ($T:ty, $XGEMM:path) => {
        impl XgemmScalar for $T {
            fn xgemm(
                transa: u8, transb: u8, m: i32, n: i32, k: i32, alpha: Self, a: &[Self], 
                lda: i32, b: &[Self], ldb: i32, beta: Self, c: &mut [Self], ldc: i32
            ) {
                unsafe{
                    $XGEMM(
                        transa, transb, m, n, k, alpha, a, 
                        lda, b, ldb, beta, c, ldc
                    );
                }
            }
        }
    };
}

impl_blas_gemm!(f32, sgemm);
impl_blas_gemm!(f64, dgemm);

// --------------------------------------
// ?gemv : matrix vector multiply (general shape)
// --------------------------------------

pub trait XgemvScalar: Sized {
    fn xgemv(
        trans: u8, m: i32, n: i32, alpha: Self, a: &[Self], lda: i32, 
        x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32
    );
}


macro_rules! impl_blas_gemv {
    ($T:ty, $XGEMV:path) => {
        impl XgemvScalar for $T {
            fn xgemv(
                trans: u8, m: i32, n: i32, alpha: Self, a: &[Self], lda: i32, 
                x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32
            ) {
                unsafe{
                    $XGEMV(
                        trans, m, n, alpha, a, lda, x, incx, beta, y, incy
                    );
                }
            }
        }
    };
}

impl_blas_gemv!(f32, sgemv);
impl_blas_gemv!(f64, dgemv);


// --------------------------------------
// ?symv : matrix vector multiply (symmetric)
// --------------------------------------

pub trait XsymvScalar: Sized {
    fn xsymv(
        uplo: u8, n: i32, alpha: Self, a: &[Self], lda: i32, 
        x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32
    );
}


macro_rules! impl_blas_gsymv {
    ($T:ty, $XSYMV:path) => {
        impl XsymvScalar for $T {
            fn xsymv(
                uplo: u8, n: i32, alpha: Self, a: &[Self], lda: i32, 
                x: &[Self], incx: i32, beta: Self, y: &mut [Self], incy: i32
            ) {
                unsafe{
                    $XSYMV(
                        uplo, n, alpha, a, lda, x, incx, beta, y, incy
                    );
                }
            }
        }
    };
}

impl_blas_gsymv!(f32, ssymv);
impl_blas_gsymv!(f64, dsymv);


// --------------------------------------
// ?syrk : symmetric rank k update
// --------------------------------------

pub trait XsyrkScalar: Sized {
    fn xsyrk(
        uplo: u8, trans: u8, n: i32, k: i32, alpha: Self, 
        a: &[Self], lda: i32, beta: Self, c: &mut [Self], ldc: i32
    );
}


macro_rules! impl_blas_gsyrk {
    ($T:ty, $XSYRK:path) => {
        impl XsyrkScalar for $T {
            fn xsyrk(
                uplo: u8, trans: u8, n: i32, k: i32, alpha: Self, 
                a: &[Self], lda: i32, beta: Self, c: &mut [Self], ldc: i32
            ) {
                unsafe{
                    $XSYRK(
                        uplo, trans, n, k, alpha, a, lda, beta, c, ldc
                    );
                }
            }
        }
    };
}

impl_blas_gsyrk!(f32, ssyrk);
impl_blas_gsyrk!(f64, dsyrk);