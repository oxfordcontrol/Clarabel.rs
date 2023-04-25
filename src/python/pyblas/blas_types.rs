#![allow(clippy::upper_case_acronyms)]
use libc::{c_char, c_double, c_float, c_int};

pub struct PyBlasPointers {
    pub ddot_: DDOT,
    pub sdot_: SDOT,
    pub dgemm_: DGEMM,
    pub sgemm_: SGEMM,
    pub dgemv_: DGEMV,
    pub sgemv_: SGEMV,
    pub dsymv_: DSYMV,
    pub ssymv_: SSYMV,
    pub dsyrk_: DSYRK,
    pub ssyrk_: SSYRK,
    pub dsyr2k_: DSYR2K,
    pub ssyr2k_: SSYR2K,
}

type DDOT = extern "C" fn(
    n: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    y: *const c_double,
    incy: *const c_int,
) -> c_double;

type SDOT = extern "C" fn(
    n: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    y: *const c_float,
    incy: *const c_int,
) -> c_float;

type DGEMM = extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    b: *const c_double,
    ldb: *const c_int,
    beta: *const c_double,
    c: *mut c_double,
    ldc: *const c_int,
);

type SGEMM = extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const c_int,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    b: *const c_float,
    ldb: *const c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: *const c_int,
);

type DGEMV = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    beta: *const c_double,
    y: *mut c_double,
    incy: *const c_int,
);

type SGEMV = extern "C" fn(
    trans: *const c_char,
    m: *const c_int,
    n: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: *const c_int,
);

type DSYMV = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    x: *const c_double,
    incx: *const c_int,
    beta: *const c_double,
    y: *mut c_double,
    incy: *const c_int,
);

type SSYMV = extern "C" fn(
    uplo: *const c_char,
    n: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    x: *const c_float,
    incx: *const c_int,
    beta: *const c_float,
    y: *mut c_float,
    incy: *const c_int,
);

type DSYRK = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    beta: *const c_double,
    c: *mut c_double,
    ldc: *const c_int,
);

type SSYRK = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: *const c_int,
);

type DSYR2K = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_double,
    a: *const c_double,
    lda: *const c_int,
    b: *const c_double,
    ldb: *const c_int,
    beta: *const c_double,
    c: *mut c_double,
    ldc: *const c_int,
);

type SSYR2K = extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const c_int,
    k: *const c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: *const c_int,
    b: *const c_float,
    ldb: *const c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: *const c_int,
);
