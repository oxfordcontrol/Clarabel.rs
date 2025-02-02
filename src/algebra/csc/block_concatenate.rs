use crate::algebra::hvcat_dim_check;
use crate::algebra::matrix_traits::ShapedMatrix;
use crate::algebra::MatrixConcatenationError;
use std::cmp::max;

use crate::algebra::{BlockConcatenate, CscMatrix, FloatT, MatrixShape};

impl<T> BlockConcatenate for CscMatrix<T>
where
    T: FloatT,
{
    fn hcat(A: &Self, B: &Self) -> Result<Self, MatrixConcatenationError> {
        Self::hvcat(&[&[A, B]])
    }

    fn vcat(A: &Self, B: &Self) -> Result<Self, MatrixConcatenationError> {
        Self::hvcat(&[&[A], &[B]])
    }

    //PJG: This might be modifiable to allow Adjoint and Symmetric
    //inputs as well.
    fn blockdiag(mats: &[&Self]) -> Result<Self, MatrixConcatenationError> {
        if mats.is_empty() {
            return Err(MatrixConcatenationError::IncompatibleDimension);
        }

        let mut nrows = 0;
        let mut ncols = 0;
        let mut nnzM = 0;
        for mat in mats {
            nrows += mat.nrows();
            ncols += mat.ncols();
            nnzM += mat.nnz();
        }
        let mut M = CscMatrix::<T>::spalloc((nrows, ncols), nnzM);

        // assemble the column counts
        M.colptr.fill(0);

        let mut nextcol = 0;
        for mat in mats {
            M.colcount_block(mat, nextcol, MatrixShape::N);
            nextcol += mat.ncols();
        }

        M.colcount_to_colptr();

        //PJG: create fake data map showing where the
        //entries go.   Probably this should be an Option
        //instead, but that requires rewriting some of the
        //KKT assembly code.

        // unwrap is fine since this is unreachable for empty input
        let dummylength = mats.iter().map(|m| m.nnz()).max().unwrap();
        let mut dummymap = vec![0; dummylength];

        // fill in data and rebuild colptr
        let mut nextrow = 0;
        let mut nextcol = 0;
        for mat in mats {
            M.fill_block(mat, &mut dummymap, nextrow, nextcol, MatrixShape::N);
            nextrow += mat.nrows();
            nextcol += mat.ncols();
        }

        M.backshift_colptrs();

        Ok(M)
    }

    fn hvcat(mats: &[&[&Self]]) -> Result<Self, MatrixConcatenationError> {
        // check for consistent block dimensions
        hvcat_dim_check(mats)?;

        // dimensions are consistent and nonzero, so count
        // total rows and columns by counting along the border
        let nrows = mats.iter().map(|blockrow| blockrow[0].nrows()).sum();
        let ncols = mats[0].iter().map(|topblock| topblock.ncols()).sum();

        let mut nnzM = 0;
        let mut maxblocknnz = 0; // for dummy mapping below
        for &blockrow in mats {
            for mat in blockrow {
                let blocknnz = mat.nnz();
                maxblocknnz = max(maxblocknnz, blocknnz);
                nnzM += blocknnz;
            }
        }

        let mut M = CscMatrix::<T>::spalloc((nrows, ncols), nnzM);

        // assemble the column counts
        M.colptr.fill(0);
        let mut currentcol = 0;
        for i in 0..mats[0].len() {
            for blockrow in mats {
                M.colcount_block(blockrow[i], currentcol, MatrixShape::N);
            }
            currentcol += mats[0][i].ncols();
        }

        M.colcount_to_colptr();

        //PJG: create fake data maps showing where the
        //entries go.   Probably this should be an Option
        //instead, but that requires rewriting some of the
        //KKT assembly code
        let mut dummymap = vec![0; maxblocknnz];

        // fill in data and rebuild colptr
        let mut currentcol = 0;
        for i in 0..mats[0].len() {
            let mut currentrow = 0;
            for blockrow in mats {
                M.fill_block(
                    blockrow[i],
                    &mut dummymap,
                    currentrow,
                    currentcol,
                    MatrixShape::N,
                );
                currentrow += blockrow[i].nrows();
            }
            currentcol += mats[0][i].ncols();
        }

        M.backshift_colptrs();

        Ok(M)
    }
}

#[test]
fn test_dense_concatenate() {
    let A = CscMatrix::from(&[
        [1., 3.], //
        [2., 4.], //
    ]);
    let B = CscMatrix::from(&[
        [5., 7.], //
        [6., 8.], //
    ]);

    // horizontal
    let C = CscMatrix::hcat(&A, &B).unwrap();
    let Ctest = CscMatrix::from(&[
        [1., 3., 5., 7.], //
        [2., 4., 6., 8.], //
    ]);

    assert_eq!(C, Ctest);

    // vertical
    let C = CscMatrix::vcat(&A, &B).unwrap();
    let Ctest = CscMatrix::from(&[
        [1., 3.], //
        [2., 4.], //
        [5., 7.], //
        [6., 8.], //
    ]);
    assert_eq!(C, Ctest);

    // 2 x 2 block
    let C = CscMatrix::hvcat(&[&[&A, &B], &[&B, &A]]).unwrap();
    let Ctest = CscMatrix::from(&[
        [1., 3., 5., 7.], //
        [2., 4., 6., 8.], //
        [5., 7., 1., 3.], //
        [6., 8., 2., 4.], //
    ]);
    assert_eq!(C, Ctest);

    // block diagonal
    let C = CscMatrix::blockdiag(&[&A, &B]).unwrap();
    let Ctest = CscMatrix::from(&[
        [1., 3., 0., 0.], //
        [2., 4., 0., 0.], //
        [0., 0., 5., 7.], //
        [0., 0., 6., 8.], //
    ]);
    assert_eq!(C, Ctest);
}
