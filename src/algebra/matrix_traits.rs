#![allow(non_snake_case)]

use std::ops::Index;

use crate::algebra::MatrixShape;

pub trait ShapedMatrix {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn size(&self) -> (usize, usize);
    fn shape(&self) -> MatrixShape;
    fn is_square(&self) -> bool;
}

//NB: the concrete dense type is just called "Matrix".  The "DenseMatrix" trait
//is implemented on both Matrix<T> and Adjoint<'a,Matrix<T>> to allow for direct
//indexing of values in either format.   This follows the Julia naming convention
//for similar types.

pub trait DenseMatrix: ShapedMatrix + Index<(usize, usize)> {
    type T;
    fn index_linear(&self, idx: (usize, usize)) -> usize;
    fn data(&self) -> &[Self::T];
}

pub trait BlockConcatenate {
    /// horizontal matrix concatenation:
    /// ```text
    /// C = [A B]
    /// ```
    /// # Panics
    /// Panics if row dimensions are incompatible

    fn hcat(A: &Self, B: &Self) -> Self;

    /// vertical matrix concatenation:
    /// ```text
    /// C = [ A ]
    ///     [ B ]
    /// ```
    ///
    /// # Panics
    /// Panics if column dimensions are incompatible

    fn vcat(A: &Self, B: &Self) -> Self;
}
