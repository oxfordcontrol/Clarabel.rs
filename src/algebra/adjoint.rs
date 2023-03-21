/// Adjoint of a matrix
use crate::algebra::{Adjoint, MatrixShape, ShapedMatrix};

impl<'a, M> ShapedMatrix for Adjoint<'a, M>
where
    M: ShapedMatrix,
{
    fn nrows(&self) -> usize {
        self.src.ncols()
    }
    fn ncols(&self) -> usize {
        self.src.nrows()
    }
    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::T
    }
    fn is_square(&self) -> bool {
        self.src.is_square()
    }
}
