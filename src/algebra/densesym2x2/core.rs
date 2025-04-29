#![allow(non_snake_case)]

use crate::algebra::*;
use std::ops::{Index, IndexMut};

// 2x2 Dense matrix types are restricted to the crate
// NB: Implements special matrix decomposition cases
//
// Data is stored as an array of 3 values belonging
// the upper triangle of a 3x3 matrix.   Lower triangle
// is assumed symmetric.    

// NB: this implementation is very limited, but there 
// is still a lot of redundant code with the 3x3 case.
// 2x2 and 3x3 types could probably be simplified

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenseMatrixSym2<T> {
    pub data: [T; 3],
}

impl<T: FloatT> Index<(usize, usize)> for DenseMatrixSym2<T> {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[Self::index_linear(idx)]
    }
}

impl<T: FloatT> IndexMut<(usize, usize)> for DenseMatrixSym2<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.data[Self::index_linear(idx)]
    }
}



#[allow(dead_code)]
impl<T> DenseMatrixSym2<T>
where
    T: FloatT,
{
    pub fn zeros() -> Self {
        Self {
            data: [T::zero(); 3],
        }
    }

    // y = H*x
    pub fn mul(&self, y: &mut [T], x: &[T]) {
        let H = self;

        //matrix is packed triu of a 2x2, so unroll it here
        y[0] = (H.data[0] * x[0]) + (H.data[1] * x[1]);
        y[1] = (H.data[1] * x[0]) + (H.data[2] * x[1]);
    }

    //convert row col coordinate to triu index
    #[inline]
    pub const fn index_linear(idx: (usize, usize)) -> usize {
        let (r, c) = idx;
        if r < c {
            r + triangular_number(c)
        } else {
            c + triangular_number(r)
        }
    }
}

impl<T> From<DenseMatrixSym2<T>> for Matrix<T>
where
    T: FloatT,
{
    #[rustfmt::skip]
    fn from(S: DenseMatrixSym2<T>) -> Self {

        let a = S.data[0];
        let b = S.data[1];
        let c = S.data[2];

        Matrix::from(&[
            [a, b], 
            [b, c]])
    }
}

impl<M,T> From<Symmetric<'_,M>> for DenseMatrixSym2<T>
where
    T: FloatT,
    M: DenseMatrix<T>,
{
    fn from(S: Symmetric<'_,M>) -> Self {

        debug_assert!(
            S.src.size() == (2,2),
            "Matrix must be 2,2 to convert to DenseMatrixSym3",
        );

        let vals = S.src.data();  
        match S.uplo{
            MatrixTriangle::Triu => {
                //read from upper triangle
                let a = vals[0];
                let b = vals[2];
                let c = vals[3];

                //pack upper triangle
                Self {data: [a, b, c]}
            },
            MatrixTriangle::Tril => {
                //read from lower triangle
                let a = vals[0];
                let b = vals[1];
                let c = vals[3];

                //pack upper triangle
                Self {data: [a, b, c]}
            }
        }
    }
}