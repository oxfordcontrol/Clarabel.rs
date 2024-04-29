/// Adjoint of a matrix
use crate::algebra::*;

/// Borrowed data slice reshaped into a matrix.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BorrowedMatrix<'a, T> {
    /// number of rows
    pub m: usize,
    ///number of columns
    pub n: usize,
    ///borrowed data
    pub data: &'a [T],
}

/// Borrowed mutable data slice reshaped into a matrix.
#[derive(Debug, PartialEq)]
pub(crate) struct BorrowedMatrixMut<'a, T> {
    /// number of rows
    pub m: usize,
    ///number of columns
    pub n: usize,
    ///borrowed data
    pub data: &'a mut [T],
}

impl<'a, T> ShapedMatrix for BorrowedMatrix<'a, T>
where
    T: FloatT,
{
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
}

impl<'a, T> ShapedMatrix for BorrowedMatrixMut<'a, T>
where
    T: FloatT,
{
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
}

// PJG: this is redundant with the Matrix implementation
// of the same functions.   Operations in Matrix that
// are for accessing and selection data should be implemented
// on the DenseMatrix trait, with mutable ones on DenseMatrixMut.
// This is a temporary measure to allow for the reshaped to
// be multiplied columnwise in the SVD
impl<'a, T> BorrowedMatrixMut<'a, T> {
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

impl<'a, T> BorrowedMatrix<'a, T>
where
    T: FloatT,
{
    pub fn from_slice(data: &'a [T], m: usize, n: usize) -> Self {
        Self { data, m, n }
    }
}

impl<'a, T> BorrowedMatrixMut<'a, T>
where
    T: FloatT,
{
    pub fn from_slice_mut(data: &'a mut [T], m: usize, n: usize) -> Self {
        Self { data, m, n }
    }
}
