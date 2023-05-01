#![allow(non_snake_case)]

use crate::algebra::{
    DenseFactorizationError, FactorCholesky, FloatT, Matrix, MatrixTriangle, ShapedMatrix,
    VectorMath,
};

pub(crate) struct CholeskyEngine<T> {
    /// lower triangular factor (stored as square dense)
    pub L: Matrix<T>,
}

impl<T> CholeskyEngine<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        let L = Matrix::<T>::zeros((n, n));
        Self { L }
    }
}

impl<T> FactorCholesky for CholeskyEngine<T>
where
    T: FloatT,
{
    type T = T;
    fn cholesky(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError> {
        if A.size() != self.L.size() {
            return Err(DenseFactorizationError::IncompatibleDimension);
        }

        // standard BLAS ?potrf arguments for computing
        // cholesky decomposition
        let uplo = MatrixTriangle::Triu.as_blas_char(); // only look at triu of A
        let An = A.nrows().try_into().unwrap();
        let a = A.data_mut();
        let lda = An;
        let info = &mut 0_i32; // output info

        T::xpotrf(uplo, An, a, lda, info);

        if *info != 0 {
            return Err(DenseFactorizationError::Cholesky(*info));
        }

        // A will now have L^T in its upper triangle.
        let At = A.t();
        self.L.data_mut().set(T::zero());

        let n = self.L.nrows();
        for j in 0..n {
            for i in j..n {
                self.L[(i, j)] = At[(i, j)];
            }
        }

        Ok(())
    }
}

#[test]
fn test_cholesky() {
    use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

    #[rustfmt::skip]
    let mut S = Matrix::from(
        &[[ 8., -2., 4.],
          [-2., 12., 2.],
          [ 4.,  2., 6.]]);

    let Scopy = S.clone(); //S is corrupted after factorization

    let mut eng = CholeskyEngine::<f64>::new(3);
    assert!(eng.cholesky(&mut S).is_ok());

    let mut M = Matrix::<f64>::zeros((3, 3));
    M.mul(&eng.L, &eng.L.t(), 1.0, 0.0);

    assert!(M.data().norm_inf_diff(Scopy.data()) < 1e-8);
}
