use crate::algebra::hvcat_dim_check;
use crate::algebra::matrix_traits::ShapedMatrix;
use crate::algebra::MatrixConcatenationError;
use std::cmp::max;

use crate::algebra::{BlockConcatenate, CscMatrix, FloatT, MatrixShape};

// PJG: hcat and vcat should return option and should probably
// just call the internal hvcat function.   Needs examples, unit
// tests and make this a public interface

impl<T> BlockConcatenate for CscMatrix<T>
where
    T: FloatT,
{
    fn hcat(A: &Self, B: &Self) -> Self {
        //first check for compatible row dimensions
        assert_eq!(A.m, B.m);

        //dimensions for C = [A B];
        let nnz = A.nnz() + B.nnz();
        let m = A.m; //rows C
        let n = A.n + B.n; //cols C
        let mut C = CscMatrix::spalloc((m, n), nnz);

        //we make dummy mapping indices since we don't care
        //where the entries go.  An alternative would be to
        //modify the fill_block method to accept Option<_>
        let mut map = vec![0usize; max(A.nnz(), B.nnz())];

        //compute column counts and fill
        C.colcount_block(A, 0, MatrixShape::N);
        C.colcount_block(B, A.n, MatrixShape::N);
        C.colcount_to_colptr();

        C.fill_block(A, &mut map, 0, 0, MatrixShape::N);
        C.fill_block(B, &mut map, 0, A.n, MatrixShape::N);
        C.backshift_colptrs();

        C
    }

    fn vcat(A: &Self, B: &Self) -> Self {
        //first check for compatible column dimensions
        assert_eq!(A.n, B.n);

        //dimensions for C = [A; B];
        let nnz = A.nnz() + B.nnz();
        let m = A.m + B.m; //rows C
        let n = A.n; //cols C
        let mut C = CscMatrix::spalloc((m, n), nnz);

        //we make dummy mapping indices since we don't care
        //where the entries go.  An alternative would be to
        //modify the fill_block method to accept Option<_>
        let mut map = vec![0usize; max(A.nnz(), B.nnz())];

        //compute column counts and fill
        C.colcount_block(A, 0, MatrixShape::N);
        C.colcount_block(B, 0, MatrixShape::N);
        C.colcount_to_colptr();

        C.fill_block(A, &mut map, 0, 0, MatrixShape::N);
        C.fill_block(B, &mut map, A.m, 0, MatrixShape::N);
        C.backshift_colptrs();
        C
    }

    /// Horizontal and vertical concatentation of matrix blocks
    /// Errors if given data of incompatible dimensions
    ///
    ///
    //PJG: This should allow transposes as well.   In dire need of example
    //and units tests

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

//PJG: hvcat and blockdiag unittests here
