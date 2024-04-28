/// Adjoint of a matrix
use crate::algebra::*;

impl<'a, T> ShapedMatrix for ReshapedMatrix<'a, T>
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

impl<'a, T> ShapedMatrix for ReshapedMatrixMut<'a, T>
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
