use thiserror::Error;

/// Error type returned by sparse matrix user input checkers, e.g. [`check_format`](crate::algebra::CscMatrix::check_format) utility.
#[derive(Error, Debug)]
pub enum SparseFormatError {
    #[error("Matrix dimension fields and/or array lengths are incompatible")]
    IncompatibleDimension,
    #[error("Data is not sorted by row index within each column")]
    BadRowOrdering,
    #[error("Row value exceeds the matrix row dimension")]
    BadRowval,
    #[error("Bad column pointer values")]
    BadColptr,
    #[error("sparsity pattern mismatch")]
    SparsityMismatch,
}

/// Error type returned by BLAS-like dense factorization routines.  Errors
/// return the internal BLAS error codes.
#[derive(Error, Debug)]
pub enum DenseFactorizationError {
    #[error("Matrix dimension fields and/or array lengths are incompatible")]
    IncompatibleDimension,
    #[error("Eigendecomposition error")]
    Eigen(i32),
    #[error("SVD error")]
    SVD(i32),
    #[error("Cholesky error")]
    Cholesky(i32),
}
