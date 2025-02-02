use thiserror::Error;

/// Error type returned by matrix concatenation operations.
#[derive(Error, Debug)]
pub enum MatrixConcatenationError {
    #[error("Incompatible dimensions")]
    /// Indicates inputs have incompatible dimension
    IncompatibleDimension,
}

#[derive(Error, Debug)]
/// Error type returned by sparse matrix assembly operations.
pub enum SparseFormatError {
    /// Matrix dimension fields and/or array lengths are incompatible
    #[error("Matrix dimension fields and/or array lengths are incompatible")]
    IncompatibleDimension,
    /// Data is not sorted by row index within each column
    #[error("Data is not sorted by row index within each column")]
    BadRowOrdering,
    #[error("Row value exceeds the matrix row dimension")]
    /// Row value exceeds the matrix row dimension
    BadRowval,
    #[error("Bad column pointer values")]
    /// Matrix column pointer values are defective
    BadColptr,
    #[error("sparsity pattern mismatch")]
    /// Operation on matrices that have mismatching sparsity patterns
    SparsityMismatch,
}

/// Error type returned by BLAS-like dense factorization routines.  Errors
/// return the internal BLAS error codes.
#[allow(clippy::upper_case_acronyms)]
#[allow(dead_code)]
#[derive(Error, Debug)]
pub(crate) enum DenseFactorizationError {
    #[error("Matrix dimension fields and/or array lengths are incompatible")]
    IncompatibleDimension,
    #[error("Eigendecomposition error")]
    Eigen(i32),
    #[error("SVD error")]
    SVD(i32),
    #[error("Cholesky error")]
    Cholesky(i32),
    #[error("LU error")]
    LU(i32),
}
