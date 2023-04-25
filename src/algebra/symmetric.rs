/// Adjoint of a matrix
use crate::algebra::{MatrixShape, ShapedMatrix, Symmetric};

impl<'a, M> ShapedMatrix for Symmetric<'a, M>
where
    M: ShapedMatrix,
{
    fn nrows(&self) -> usize {
        self.src.nrows()
    }
    fn ncols(&self) -> usize {
        self.src.ncols()
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
}
