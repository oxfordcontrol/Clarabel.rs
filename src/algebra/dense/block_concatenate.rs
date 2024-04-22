#![allow(non_snake_case)]
use std::ops::Index;

use crate::algebra::{
    hvcat_dim_check, BlockConcatenate, DenseMatrix, FloatT, Matrix, MatrixConcatenationError,
    ShapedMatrix,
};

impl<T> BlockConcatenate for Matrix<T>
where
    T: FloatT,
{
    fn hcat(A: &Self, B: &Self) -> Self {
        //first check for compatible row dimensions
        assert_eq!(A.m, B.m);

        //dimensions for C = [A B];
        let m = A.m; //rows
        let n = A.n + B.n; //cols s
        let mut data = A.data.clone();
        data.extend(&B.data);
        Self { m, n, data }
    }

    fn vcat(A: &Self, B: &Self) -> Self {
        //first check for compatible column dimensions
        assert_eq!(A.n, B.n);

        //dimensions for C = [A; B];
        let m = A.m + B.m; //rows C
        let n = A.n; //cols C
        let mut data = Vec::with_capacity(m * n);

        for col in 0..A.ncols() {
            data.extend(A.col_slice(col));
            data.extend(B.col_slice(col));
        }
        Self { m, n, data }
    }

    fn hvcat(mats: &[&[&Self]]) -> Result<Self, MatrixConcatenationError> {
        // check for consistent block dimensions
        hvcat_dim_check(mats)?;

        // dimensions are consistent and nonzero, so count
        // total rows and columns by counting along the border
        let nrows = mats.iter().map(|blockrow| blockrow[0].nrows()).sum();
        let ncols = mats[0].iter().map(|topblock| topblock.ncols()).sum();

        let mut data = Vec::with_capacity(nrows * ncols);

        for blockcol in 0..mats[0].len() {
            //every matrix in each block-column should have the same
            //number of columns
            for col in 0..mats[blockcol][0].ncols() {
                for blockrow in 0..mats.len() {
                    let block = mats[blockrow][blockcol];
                    data.extend(block.col_slice(col));
                }
            }
        }
        Ok(Self::new((nrows, ncols), data))
    }

    fn blockdiag(mats: &[&Self]) -> Result<Self, MatrixConcatenationError> {
        if mats.is_empty() {
            return Err(MatrixConcatenationError::IncompatibleDimension);
        }

        let nrows = mats.iter().map(|mat| mat.nrows()).sum();
        let ncols = mats.iter().map(|mat| mat.ncols()).sum();

        let mut M = Self::zeros((nrows, ncols));

        // top left entry index of next block
        let mut blockrow = 0;
        let mut blockcol = 0;

        for &block in mats {
            for col in 0..block.ncols() {
                let startidx = M.index_linear((blockrow, blockcol + col));
                let range = startidx..(startidx + block.nrows());
                M.data[range].copy_from_slice(block.col_slice(col));
            }
            blockrow += block.nrows();
            blockcol += block.ncols();
        }
        Ok(M)
    }
}

#[test]
fn test_dense_concatenate() {
    let A = Matrix::from(&[
        [1., 3.], //
        [2., 4.], //
    ]);
    let B = Matrix::from(&[
        [5., 7.], //
        [6., 8.], //
    ]);

    let C = Matrix::hcat(&A, &B);

    let Ctest = Matrix::from(&[
        [1., 3., 5., 7.], //
        [2., 4., 6., 8.], //
    ]);

    assert_eq!(C, Ctest);

    let C = Matrix::vcat(&A, &B);

    let Ctest = Matrix::from(&[
        [1., 3.], //
        [2., 4.], //
        [5., 7.], //
        [6., 8.], //
    ]);

    assert_eq!(C, Ctest);
}

//PJG: hvcat and blockdiag unittests here
