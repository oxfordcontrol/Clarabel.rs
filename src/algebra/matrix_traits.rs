#![allow(non_snake_case)]

use std::ops::Index;

use crate::algebra::MatrixConcatenationError;
use crate::algebra::MatrixShape;

pub(crate) trait ShapedMatrix {
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    #[allow(dead_code)] //PJG: not currently used anywhere
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

/// PJG: replace panics here with Error types.  Add documentation
/// and examples for blockdiag and hvcat

/// Blockwise matrix concatenation
pub trait BlockConcatenate: Sized {
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

    fn hvcat(mats: &[&[&Self]]) -> Result<Self, MatrixConcatenationError>;

    fn blockdiag(mats: &[&Self]) -> Result<Self, MatrixConcatenationError>;
}

pub(crate) fn hvcat_dim_check<MAT: ShapedMatrix>(
    mats: &[&[&MAT]],
) -> Result<(), MatrixConcatenationError> {
    // error if no blocks
    if mats.is_empty() || mats[0].is_empty() {
        return Err(MatrixConcatenationError::IncompatibleDimension);
    };

    // error unless every block row has the same number of blocks
    let len0 = mats[0].len();
    for mat in mats.iter().skip(1) {
        if mat.len() != len0 {
            return Err(MatrixConcatenationError::IncompatibleDimension);
        }
    }

    // check for block dimensional consistency across block row and columns

    //row checks
    for blockrow in mats {
        let rows = blockrow[0].nrows();
        for mat in blockrow.iter().skip(1) {
            if mat.nrows() != rows {
                return Err(MatrixConcatenationError::IncompatibleDimension);
            }
        }
    }

    // column checks
    for (blockcol, topblock) in mats[0].iter().enumerate() {
        let cols = topblock.ncols();
        for matrow in mats.iter().skip(1) {
            if matrow[blockcol].ncols() != cols {
                return Err(MatrixConcatenationError::IncompatibleDimension);
            }
        }
    }

    Ok(())
}
