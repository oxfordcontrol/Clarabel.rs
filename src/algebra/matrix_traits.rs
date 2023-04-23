#![allow(non_snake_case)]

use std::ops::Index;

use crate::algebra::MatrixShape;

pub(crate) trait ShapedMatrix {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn shape(&self) -> MatrixShape;
    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }
}

//NB: the concrete dense type is just called "Matrix".  The "DenseMatrix" trait
//is implemented on Matrix, Adjoint and ReshapedMatrix to allow for indexing
//of values in any of those format.   This follows the Julia naming convention
//for similar types.
pub(crate) trait DenseMatrix: ShapedMatrix + Index<(usize, usize)> {
    type T;
    fn index_linear(&self, idx: (usize, usize)) -> usize;
    fn data(&self) -> &[Self::T];
}

/// Blockwise matrix concatenation
pub trait BlockConcatenate {
    /// horizontal matrix concatenation
    ///
    /// ```text
    /// C = [A B]
    /// ```
    /// # Panics
    /// Panics if row dimensions are incompatible

    fn hcat(A: &Self, B: &Self) -> Self;

    /// vertical matrix concatenation
    ///
    /// ```text
    /// C = [ A ]
    ///     [ B ]
    /// ```
    ///
    /// # Panics
    /// Panics if column dimensions are incompatible

    fn vcat(A: &Self, B: &Self) -> Self;
}
