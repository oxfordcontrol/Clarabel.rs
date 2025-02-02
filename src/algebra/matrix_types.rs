use crate::algebra::ShapedMatrix;

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
/// Symmetric view of a matrix.   Only the upper
/// triangle of the source matrix will be referenced.
#[derive(Debug, Clone, PartialEq)]
pub struct Symmetric<'a, M> {
    pub src: &'a M,
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
