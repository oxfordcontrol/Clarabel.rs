#![allow(clippy::upper_case_acronyms)]
use libc::{c_char, c_int};

pub struct PyLapackPointers {
    pub dsyevr_: DSYEVR,
    pub ssyevr_: SSYEVR,
    pub dpotrf_: DPOTRF,
    pub spotrf_: SPOTRF,
    pub dpotrs_: DPOTRS,
    pub spotrs_: SPOTRS,
    pub dgesdd_: DGESDD,
    pub sgesdd_: SGESDD,
    pub dgesvd_: DGESVD,
    pub sgesvd_: SGESVD,
    pub dgesv_: DGESV,
    pub sgesv_: SGESV,
}

type DSYEVR = extern "C" fn(
    jobz: *const c_char,
    range: *const c_char,
    uplo: *const c_char,
    n: *const c_int,
    A: *mut f64,
    lda: *const c_int,
    vl: *const f64,
    vu: *const f64,
    il: *const c_int,
    iu: *const c_int,
    abstol: *const f64,
    m: *mut c_int,
    W: *mut f64,
    Z: *mut f64,
    ldz: *const c_int,
    ISUPPZ: *mut c_int,
    work: *mut f64,
    lwork: *const c_int,
    iwork: *mut c_int,
    liwork: *const c_int,
    info: *mut c_int,
);

type SSYEVR = extern "C" fn(
    jobz: *const c_char,
    range: *const c_char,
    uplo: *const c_char,
    n: *const c_int,
    A: *mut f32,
    lda: *const c_int,
    vl: *const f32,
    vu: *const f32,
    il: *const c_int,
    iu: *const c_int,
    abstol: *const f32,
    m: *mut c_int,
    W: *mut f32,
    Z: *mut f32,
    ldz: *const c_int,
    ISUPPZ: *mut c_int,
    work: *mut f32,
    lwork: *const c_int,
    iwork: *mut c_int,
    liwork: *const c_int,
    info: *mut c_int,
);

type DPOTRF = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    A: *mut f64,
    lda: *const c_int,
    info: *mut c_int,
);

type SPOTRF = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    A: *mut f32,
    lda: *const c_int,
    info: *mut c_int,
);

type DPOTRS = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    nrhs: *const c_int,
    A: *const f64,
    lda: *const c_int,
    B: *mut f64,
    ldb: *const c_int,
    info: *mut c_int,
);

type SPOTRS = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    nrhs: *const c_int,
    A: *const f32,
    lda: *const c_int,
    B: *mut f32,
    ldb: *const c_int,
    info: *mut c_int,
);

type DGESDD = extern "C" fn(
    jobz: *const c_char,
    m: *const c_int,
    n: *const c_int,
    A: *mut f64,
    lda: *const c_int,
    S: *mut f64,
    U: *mut f64,
    ldu: *const c_int,
    VT: *mut f64,
    ldvt: *const c_int,
    work: *mut f64,
    lwork: *const c_int,
    iwork: *mut c_int,
    info: *mut c_int,
);

type SGESDD = extern "C" fn(
    jobz: *const c_char,
    m: *const c_int,
    n: *const c_int,
    A: *mut f32,
    lda: *const c_int,
    S: *mut f32,
    U: *mut f32,
    ldu: *const c_int,
    VT: *mut f32,
    ldvt: *const c_int,
    work: *mut f32,
    lwork: *const c_int,
    iwork: *mut c_int,
    info: *mut c_int,
);

type DGESVD = extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const c_int,
    n: *const c_int,
    A: *mut f64,
    lda: *const c_int,
    S: *mut f64,
    U: *mut f64,
    ldu: *const c_int,
    VT: *mut f64,
    ldvt: *const c_int,
    work: *mut f64,
    lwork: *const c_int,
    info: *mut c_int,
);

type SGESVD = extern "C" fn(
    jobu: *const c_char,
    jobvt: *const c_char,
    m: *const c_int,
    n: *const c_int,
    A: *mut f32,
    lda: *const c_int,
    S: *mut f32,
    U: *mut f32,
    ldu: *const c_int,
    VT: *mut f32,
    ldvt: *const c_int,
    work: *mut f32,
    lwork: *const c_int,
    info: *mut c_int,
);

type DGESV = extern "C" fn(
    n: *const c_int,
    nrhs: *const c_int,
    A: *mut f64,
    lda: *const c_int,
    ipiv: *mut c_int,
    B: *mut f64,
    ldb: *const c_int,
    info: *mut c_int,
);

type SGESV = extern "C" fn(
    n: *const c_int,
    nrhs: *const c_int,
    A: *mut f32,
    lda: *const c_int,
    ipiv: *mut c_int,
    B: *mut f32,
    ldb: *const c_int,
    info: *mut c_int,
);
