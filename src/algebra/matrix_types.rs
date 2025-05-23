use crate::algebra::ShapedMatrix;

use super::TriangularMatrixChecks;

/// Matrix shape marker for triangular matrices
#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum MatrixTriangle {
    /// Upper triangular matrix
    Triu,
    /// Lower triangular matrix
    Tril,
}

/// Matrix triangular form marker
impl MatrixTriangle {
    /// convert to u8 character for BLAS calls
    #[cfg(feature = "sdp")]
    pub fn as_blas_char(&self) -> u8 {
        match self {
            MatrixTriangle::Triu => b'U',
            MatrixTriangle::Tril => b'L',
        }
    }
    /// transpose
    #[allow(dead_code)]
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
    #[cfg(feature = "sdp")]
    pub fn as_blas_char(&self) -> u8 {
        match self {
            MatrixShape::N => b'N',
            MatrixShape::T => b'T',
        }
    }
    /// transpose
    #[allow(dead_code)]
    pub fn t(&self) -> Self {
        match self {
            MatrixShape::N => MatrixShape::T,
            MatrixShape::T => MatrixShape::N,
        }
    }
}

//-------------------------------------
// Adjoint and Symmetric matrix views

/// Adjoint of a matrix
#[derive(Debug, Clone, PartialEq)]
pub struct Adjoint<'a, M> {
    pub src: &'a M,
}
/// Symmetric view of a matrix.   
#[derive(Debug, Clone, PartialEq)]
pub struct Symmetric<'a, M> {
    pub src: &'a M,
    pub uplo: MatrixTriangle,
}

#[allow(dead_code)]
impl<M> Symmetric<'_, M>
where
    M: TriangularMatrixChecks,
{
    pub(crate) fn is_triu_src(&self) -> bool {
        self.uplo == MatrixTriangle::Triu
    }
    pub(crate) fn is_tril_src(&self) -> bool {
        self.uplo == MatrixTriangle::Tril
    }
}

impl<M> ShapedMatrix for Adjoint<'_, M>
where
    M: ShapedMatrix,
{
    fn size(&self) -> (usize, usize) {
        (self.src.ncols(), self.src.nrows())
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::T
    }
}

impl<M> ShapedMatrix for Symmetric<'_, M>
where
    M: ShapedMatrix,
{
    fn size(&self) -> (usize, usize) {
        (self.src.ncols(), self.src.nrows())
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
}
