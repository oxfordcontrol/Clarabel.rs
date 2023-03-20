// All internal matrix representations in the default
// solver and math implementations are in standard
// compressed sparse column format, as is the API.

/// Matrix shape marker for triangular matrices
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum MatrixTriangle {
    /// Upper triangular matrix
    Triu,
    /// Lower triangular matrix
    Tril,
}

/// Matrix triangular form marker
impl MatrixTriangle {
    /// convert to u8 character for BLAS calls
    pub fn as_blas_char(&self) -> u8 {
        match self {
            MatrixTriangle::Triu => b'U',
            MatrixTriangle::Tril => b'L',
        }
    }
    /// transpose
    pub fn t(&self) -> Self {
        match self {
            MatrixTriangle::Triu => MatrixTriangle::Tril,
            MatrixTriangle::Tril => MatrixTriangle::Triu,
        }
    }
}

/// Matrix orientation marker
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MatrixShape {
    /// Normal matrix orientation
    N,
    /// Transposed matrix orientation
    T,
}

impl MatrixShape {
    /// convert to u8 character for BLAS calls
    pub fn as_blas_char(&self) -> u8 {
        match self {
            MatrixShape::N => b'N',
            MatrixShape::T => b'T',
        }
    }
    /// transpose
    pub fn t(&self) -> Self {
        match self {
            MatrixShape::N => MatrixShape::T,
            MatrixShape::T => MatrixShape::N,
        }
    }
}

/// Dense matrix in column major format
///
/// __Example usage__ : To construct the 3 x 2 matrix
/// ```text
/// A = [1.  3.  5.]
///     [2.  0.  6.]
///     [0.  4.  7.]
/// ```
///
/// ```
/// # use clarabel::algebra::CscMatrix;
///
/// // PJG: Insert example here
///
/// ```

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T = f64> {
    /// number of rows
    pub m: usize,
    ///number of columns
    pub n: usize,
    /// vector of data in column major formmat
    pub data: Vec<T>,
}

/// Sparse matrix in standard Compressed Sparse Column (CSC) format
///
/// __Example usage__ : To construct the 3 x 2 matrix
/// ```text
/// A = [1.  3.  5.]
///     [2.  0.  6.]
///     [0.  4.  7.]
/// ```
///
/// ```
/// # use clarabel::algebra::CscMatrix;
///
/// let A : CscMatrix<f64> = CscMatrix::new(
///    3,                                // m
///    3,                                // n
///    vec![0, 2, 4, 7],                 //colptr
///    vec![0, 1, 0, 2, 0, 1, 2],        //rowval
///    vec![1., 2., 3., 4., 5., 6., 7.], //nzval
///  );
///
/// // optional correctness check
/// assert!(A.check_format().is_ok());
///
/// ```
///

#[derive(Debug, Clone, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct Adjoint<'a, M> {
    pub src: &'a M,
}
