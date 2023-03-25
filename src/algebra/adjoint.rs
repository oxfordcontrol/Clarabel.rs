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
    fn shape(&self) -> MatrixShape {
        MatrixShape::T
    }
}
