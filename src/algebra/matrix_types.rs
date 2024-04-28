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

/// Adjoint of a matrix
#[derive(Debug, Clone, PartialEq)]
pub struct Adjoint<'a, M> {
    pub src: &'a M,
}
/// Symmetric view of a matrix.   Only the upper
/// triangle of the source matrix will be referenced.
#[derive(Debug, Clone, PartialEq)]
pub struct Symmetric<'a, M> {
    pub src: &'a M,
}

/// Borrowed data slice reshaped into a matrix.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ReshapedMatrix<'a, T> {
    /// number of rows
    pub m: usize,
    ///number of columns
    pub n: usize,
    ///borrowed data
    pub data: &'a [T],
}

/// Borrowed mutable data slice reshaped into a matrix.
#[derive(Debug, PartialEq)]
pub(crate) struct ReshapedMatrixMut<'a, T> {
    /// number of rows
    pub m: usize,
    ///number of columns
    pub n: usize,
    ///borrowed data
    pub data: &'a mut [T],
}

// PJG: this is redundant with the Matrix implementation
// of the same functions.   Operations in Matrix that
// are for accessing and selection data should be implemented
// on the DenseMatrix trait, with mutable ones on DenseMatrixMut.
// This is a temporary measure to allow for the reshaped to
// be multiplied columnwise in the SVD
impl<'a, T> ReshapedMatrixMut<'a, T> {
    #[allow(dead_code)]
    pub fn col_slice(&self, col: usize) -> &[T] {
        assert!(col < self.n);
        &self.data[(col * self.m)..(col + 1) * self.m]
    }

    pub fn col_slice_mut(&mut self, col: usize) -> &mut [T] {
        assert!(col < self.n);
        &mut self.data[(col * self.m)..(col + 1) * self.m]
    }
}
