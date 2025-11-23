#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]

use super::lapack_types::*;
use libc::c_char;
use std::sync::OnceLock;

static PYLAPACK: OnceLock<PyLapackPointers> = OnceLock::new();

fn get_pylapack() -> &'static PyLapackPointers {
    PYLAPACK.get_or_init(|| {
        pyo3::Python::with_gil(|py| {
            PyLapackPointers::new(py).expect("Failed to load SciPy LAPACK bindings.")
        })
    })
}

pub(crate) fn force_load() {
    // forces initialization
    let _ = get_pylapack().dsyevr_;
}

pub unsafe fn dsyevr(
    jobz: u8,
    range: u8,
    uplo: u8,
    n: i32,
    a: &mut [f64],
    lda: i32,
    vl: f64,
    vu: f64,
    il: i32,
    iu: i32,
    abstol: f64,
    m: &mut i32,
    w: &mut [f64],
    z: &mut [f64],
    ldz: i32,
    isuppz: &mut [i32],
    work: &mut [f64],
    lwork: i32,
    iwork: &mut [i32],
    liwork: i32,
    info: &mut i32,
) {
    (get_pylapack().dsyevr_)(
        &(jobz as c_char),
        &(range as c_char),
        &(uplo as c_char),
        &n,
        a.as_mut_ptr(),
        &lda,
        &vl,
        &vu,
        &il,
        &iu,
        &abstol,
        m,
        w.as_mut_ptr(),
        z.as_mut_ptr(),
        &ldz,
        isuppz.as_mut_ptr(),
        work.as_mut_ptr(),
        &lwork,
        iwork.as_mut_ptr(),
        &liwork,
        info,
    )
}

pub unsafe fn ssyevr(
    jobz: u8,
    range: u8,
    uplo: u8,
    n: i32,
    a: &mut [f32],
    lda: i32,
    vl: f32,
    vu: f32,
    il: i32,
    iu: i32,
    abstol: f32,
    m: &mut i32,
    w: &mut [f32],
    z: &mut [f32],
    ldz: i32,
    isuppz: &mut [i32],
    work: &mut [f32],
    lwork: i32,
    iwork: &mut [i32],
    liwork: i32,
    info: &mut i32,
) {
    (get_pylapack().ssyevr_)(
        &(jobz as c_char),
        &(range as c_char),
        &(uplo as c_char),
        &n,
        a.as_mut_ptr(),
        &lda,
        &vl,
        &vu,
        &il,
        &iu,
        &abstol,
        m,
        w.as_mut_ptr(),
        z.as_mut_ptr(),
        &ldz,
        isuppz.as_mut_ptr(),
        work.as_mut_ptr(),
        &lwork,
        iwork.as_mut_ptr(),
        &liwork,
        info,
    )
}

pub unsafe fn dpotrf(uplo: u8, n: i32, a: &mut [f64], lda: i32, info: &mut i32) {
    (get_pylapack().dpotrf_)(&(uplo as c_char), &n, a.as_mut_ptr(), &lda, info)
}

pub unsafe fn spotrf(uplo: u8, n: i32, a: &mut [f32], lda: i32, info: &mut i32) {
    (get_pylapack().spotrf_)(&(uplo as c_char), &n, a.as_mut_ptr(), &lda, info)
}

pub unsafe fn dpotrs(
    uplo: u8,
    n: i32,
    nrhs: i32,
    a: &[f64],
    lda: i32,
    b: &mut [f64],
    ldb: i32,
    info: &mut i32,
) {
    (get_pylapack().dpotrs_)(
        &(uplo as c_char),
        &n,
        &nrhs,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &ldb,
        info,
    )
}

pub unsafe fn spotrs(
    uplo: u8,
    n: i32,
    nrhs: i32,
    a: &[f32],
    lda: i32,
    b: &mut [f32],
    ldb: i32,
    info: &mut i32,
) {
    (get_pylapack().spotrs_)(
        &(uplo as c_char),
        &n,
        &nrhs,
        a.as_ptr(),
        &lda,
        b.as_mut_ptr(),
        &ldb,
        info,
    )
}

pub unsafe fn dgesdd(
    jobz: u8,
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    s: &mut [f64],
    u: &mut [f64],
    ldu: i32,
    vt: &mut [f64],
    ldvt: i32,
    work: &mut [f64],
    lwork: i32,
    iwork: &mut [i32],
    info: &mut i32,
) {
    (get_pylapack().dgesdd_)(
        &(jobz as c_char),
        &m,
        &n,
        a.as_mut_ptr(),
        &lda,
        s.as_mut_ptr(),
        u.as_mut_ptr(),
        &ldu,
        vt.as_mut_ptr(),
        &ldvt,
        work.as_mut_ptr(),
        &lwork,
        iwork.as_mut_ptr(),
        info,
    )
}

pub unsafe fn sgesdd(
    jobz: u8,
    m: i32,
    n: i32,
    a: &mut [f32],
    lda: i32,
    s: &mut [f32],
    u: &mut [f32],
    ldu: i32,
    vt: &mut [f32],
    ldvt: i32,
    work: &mut [f32],
    lwork: i32,
    iwork: &mut [i32],
    info: &mut i32,
) {
    (get_pylapack().sgesdd_)(
        &(jobz as c_char),
        &m,
        &n,
        a.as_mut_ptr(),
        &lda,
        s.as_mut_ptr(),
        u.as_mut_ptr(),
        &ldu,
        vt.as_mut_ptr(),
        &ldvt,
        work.as_mut_ptr(),
        &lwork,
        iwork.as_mut_ptr(),
        info,
    )
}

pub unsafe fn dgesvd(
    jobu: u8,
    jobvt: u8,
    m: i32,
    n: i32,
    a: &mut [f64],
    lda: i32,
    s: &mut [f64],
    u: &mut [f64],
    ldu: i32,
    vt: &mut [f64],
    ldvt: i32,
    work: &mut [f64],
    lwork: i32,
    info: &mut i32,
) {
    (get_pylapack().dgesvd_)(
        &(jobu as c_char),
        &(jobvt as c_char),
        &m,
        &n,
        a.as_mut_ptr(),
        &lda,
        s.as_mut_ptr(),
        u.as_mut_ptr(),
        &ldu,
        vt.as_mut_ptr(),
        &ldvt,
        work.as_mut_ptr(),
        &lwork,
        info,
    )
}

pub unsafe fn sgesvd(
    jobu: u8,
    jobvt: u8,
    m: i32,
    n: i32,
    a: &mut [f32],
    lda: i32,
    s: &mut [f32],
    u: &mut [f32],
    ldu: i32,
    vt: &mut [f32],
    ldvt: i32,
    work: &mut [f32],
    lwork: i32,
    info: &mut i32,
) {
    (get_pylapack().sgesvd_)(
        &(jobu as c_char),
        &(jobvt as c_char),
        &m,
        &n,
        a.as_mut_ptr(),
        &lda,
        s.as_mut_ptr(),
        u.as_mut_ptr(),
        &ldu,
        vt.as_mut_ptr(),
        &ldvt,
        work.as_mut_ptr(),
        &lwork,
        info,
    )
}

pub unsafe fn dgesv(
    n: i32,
    nrhs: i32,
    a: &mut [f64],
    lda: i32,
    ipiv: &mut [i32],
    b: &mut [f64],
    ldb: i32,
    info: &mut i32,
) {
    (get_pylapack().dgesv_)(
        &n,
        &nrhs,
        a.as_mut_ptr(),
        &lda,
        ipiv.as_mut_ptr(),
        b.as_mut_ptr(),
        &ldb,
        info,
    )
}

pub unsafe fn sgesv(
    n: i32,
    nrhs: i32,
    a: &mut [f32],
    lda: i32,
    ipiv: &mut [i32],
    b: &mut [f32],
    ldb: i32,
    info: &mut i32,
) {
    (get_pylapack().sgesv_)(
        &n,
        &nrhs,
        a.as_mut_ptr(),
        &lda,
        ipiv.as_mut_ptr(),
        b.as_mut_ptr(),
        &ldb,
        info,
    )
}
