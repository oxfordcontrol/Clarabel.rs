#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]
#![allow(dead_code)]

use super::blas_types::*;
use libc::c_char;
use std::sync::OnceLock;

static PYBLAS: OnceLock<PyBlasPointers> = OnceLock::new();

fn get_pyblas() -> &'static PyBlasPointers {
    PYBLAS.get_or_init(|| {
        pyo3::Python::with_gil(|py| {
            PyBlasPointers::new(py).expect("Failed to load SciPy BLAS bindings.")
        })
    })
}

pub(crate) fn force_load() {
    //forces load of the lazy static. Choice of function is arbitrary.
    let _ = get_pyblas().ddot_;
}

pub unsafe fn ddot(n: i32, x: &[f64], incx: i32, y: &[f64], incy: i32) -> f64 {
    (get_pyblas().ddot_)(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
}

pub unsafe fn sdot(n: i32, x: &[f32], incx: i32, y: &[f32], incy: i32) -> f32 {
    (get_pyblas().sdot_)(&n, x.as_ptr(), &incx, y.as_ptr(), &incy)
}

pub unsafe fn dgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    (get_pyblas().dgemm_)(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

pub unsafe fn sgemm(
    transa: u8,
    transb: u8,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    (get_pyblas().sgemm_)(
        &(transa as c_char),
        &(transb as c_char),
        &m,
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

pub unsafe fn dgemv(
    trans: u8,
    m: i32,
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    (get_pyblas().dgemv_)(
        &(trans as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

pub unsafe fn sgemv(
    trans: u8,
    m: i32,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
) {
    (get_pyblas().sgemv_)(
        &(trans as c_char),
        &m,
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

pub unsafe fn dsymv(
    uplo: u8,
    n: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    x: &[f64],
    incx: i32,
    beta: f64,
    y: &mut [f64],
    incy: i32,
) {
    (get_pyblas().dsymv_)(
        &(uplo as c_char),
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

pub unsafe fn ssymv(
    uplo: u8,
    n: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    x: &[f32],
    incx: i32,
    beta: f32,
    y: &mut [f32],
    incy: i32,
) {
    (get_pyblas().ssymv_)(
        &(uplo as c_char),
        &n,
        &alpha,
        a.as_ptr(),
        &lda,
        x.as_ptr(),
        &incx,
        &beta,
        y.as_mut_ptr(),
        &incy,
    )
}

pub unsafe fn dsyrk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    (get_pyblas().dsyrk_)(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

pub unsafe fn ssyrk(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    (get_pyblas().ssyrk_)(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

pub unsafe fn dsyr2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f64,
    a: &[f64],
    lda: i32,
    b: &[f64],
    ldb: i32,
    beta: f64,
    c: &mut [f64],
    ldc: i32,
) {
    (get_pyblas().dsyr2k_)(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}

pub unsafe fn ssyr2k(
    uplo: u8,
    trans: u8,
    n: i32,
    k: i32,
    alpha: f32,
    a: &[f32],
    lda: i32,
    b: &[f32],
    ldb: i32,
    beta: f32,
    c: &mut [f32],
    ldc: i32,
) {
    (get_pyblas().ssyr2k_)(
        &(uplo as c_char),
        &(trans as c_char),
        &n,
        &k,
        &alpha,
        a.as_ptr(),
        &lda,
        b.as_ptr(),
        &ldb,
        &beta,
        c.as_mut_ptr(),
        &ldc,
    )
}
