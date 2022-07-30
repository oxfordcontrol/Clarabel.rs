// All internal matrix representations in the default
// solver and math implementations are in standard
// compressed sparse column format, as is the API.

/// Sparse matrix in standard Compressed Sparse Column (CSC) format
///
#[derive(Debug, Clone, PartialEq)]
pub struct CscMatrix<T = f64> {
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// CSC format column pointer.   
    ///
    /// Ths field should have length `n+1`. The last entry corresponds
    /// to the the number of nonzeros and should agree with the lengths
    /// of the `rowval` and `nzval` fields.
    pub colptr: Vec<usize>,
    /// vector of row indices
    pub rowval: Vec<usize>,
    /// vector of non-zero matrix elements
    pub nzval: Vec<T>,
}

/// Matrix orientation marker
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum MatrixShape {
    /// Normal matrix orientation
    N,
    /// Transposed matrix orientation
    T,
}

/// Matrix shape marker for triangular matrices
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum MatrixTriangle {
    /// Upper triangular matrix
    Triu,
    /// Lower triangular matrix
    Tril,
}
