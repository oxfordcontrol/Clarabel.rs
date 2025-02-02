#![allow(non_snake_case)]

use crate::algebra::MatrixConcatenationError;
use crate::algebra::MatrixShape;

#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) trait ShapedMatrix {
    fn shape(&self) -> MatrixShape;
    fn size(&self) -> (usize, usize);
    fn nrows(&self) -> usize {
        self.size().0
    }
    fn ncols(&self) -> usize {
        self.size().1
    }
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }
}

/// Blockwise matrix concatenation
pub trait BlockConcatenate: Sized {
    /// horizontal matrix concatenation
    ///
    /// ```text
    /// C = [A B]
    /// ```
    ///
    /// Errors if row dimensions are incompatible
    fn hcat(A: &Self, B: &Self) -> Result<Self, MatrixConcatenationError>;

    /// vertical matrix concatenation
    ///
    /// ```text
    /// C = [ A ]
    ///     [ B ]
    /// ```
    ///
    /// Errors if column dimensions are incompatible
    fn vcat(A: &Self, B: &Self) -> Result<Self, MatrixConcatenationError>;

    /// general block concatenation
    fn hvcat(mats: &[&[&Self]]) -> Result<Self, MatrixConcatenationError>;

    /// block diagonal concatenation
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
