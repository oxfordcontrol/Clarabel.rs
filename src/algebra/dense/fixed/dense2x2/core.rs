#![allow(non_snake_case)]

use crate::algebra::{DenseMatrixN, DenseMatrixSymN, FloatT};

// 2x2 Dense matrix types are restricted to the crate
// NB: Implements special matrix decomposition cases

// NB: S = 4 here because the matrix has 2^2 elements
pub (crate) type DenseMatrix2<T> = DenseMatrixN<4, T>;

// NB: S = 3 here because the upper triangle has 3 elements
pub (crate) type DenseMatrixSym2<T> = DenseMatrixSymN<3, T>;

// hand implemented DenseMatrixSym2<T> to make sure
// everything is properly unrolled

impl<T> DenseMatrixSym2<T>
where
    T: FloatT,
{
    // y = H*x
    #[allow(dead_code)] // used in tests
    pub fn mul(&self, y: &mut [T], x: &[T]) {
        let H = self;

        //matrix is packed triu of a 2x2, so unroll it here
        y[0] = (H.data[0] * x[0]) + (H.data[1] * x[1]);
        y[1] = (H.data[1] * x[0]) + (H.data[2] * x[1]);
    }
}
