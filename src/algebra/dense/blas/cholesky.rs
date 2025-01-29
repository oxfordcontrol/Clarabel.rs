#![allow(non_snake_case)]

use crate::algebra::*;

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

    pub fn resize(&mut self, n: usize) {
        self.L.resize((n, n));
    }
}

impl<T> FactorCholesky<T> for CholeskyEngine<T>
where
    T: FloatT,
{
    fn factor<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        if A.size() != self.L.size() {
            return Err(DenseFactorizationError::IncompatibleDimension);
        }

        // ?potrf factors in place, so first copy A onto
        // our internal factor matrix L.  We reference the
        // upper triangle of A, but want a lower triangular
        // result.  LAPACK factors triu inputs to U^TU, and
        // tril inputs to LL^T, so first copy the triu part
        // of A into tril of L before we factor it
        let At = A.t();
        let n = self.L.nrows();
        for j in 0..n {
            for i in j..n {
                self.L[(i, j)] = At[(i, j)];
            }
        }

        // standard BLAS ?potrf arguments for computing
        // cholesky decomposition
        let uplo = MatrixTriangle::Tril.as_blas_char();
        let An = self.L.nrows().try_into().unwrap();
        let a = self.L.data_mut();
        let lda = An;
        let info = &mut 0_i32; // output info

        T::xpotrf(uplo, An, a, lda, info);

        if *info != 0 {
            return Err(DenseFactorizationError::Cholesky(*info));
        }

        // A will now have L^T in its upper triangle.

        Ok(())
    }

    fn solve<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>,
    {
        // standard BLAS ?potrs arguments for computing
        // post factorization triangular solve

        // Tril here since we transposed A into L before
        // factoring it
        let uplo = MatrixTriangle::Tril.as_blas_char();

        let nrhs = B.ncols().try_into().unwrap();
        let An = self.L.nrows().try_into().unwrap();
        let a = &self.L.data;
        let lda = An;
        let Bn = B.nrows().try_into().unwrap();
        let b = B.data_mut();
        let ldb = Bn;
        let info = &mut 0_i32; // output info

        T::xpotrs(uplo, An, nrhs, a, lda, b, ldb, info);

        assert_eq!(*info, 0);
    }

    fn logdet(&self) -> T {
        let mut ld = T::zero();
        let n = self.L.nrows();
        for i in 0..n {
            ld += T::ln(self.L[(i, i)]);
        }
        ld + ld
    }
}

macro_rules! generate_test_cholesky {
    ($fxx:ty, $test_name:ident, $tolfn:ident) => {
        #[test]
        fn $test_name() {
            use crate::algebra::{DenseMatrix, MultiplyGEMM, VectorMath};

            #[rustfmt::skip]
            let mut S = Matrix::<$fxx>::from(
            &[[ 8., -2., 4.],
            [-2., 12., 2.],
            [ 4.,  2., 6.]]);

            let Scopy = S.clone(); //S is corrupted after factorization

            let mut eng = CholeskyEngine::<$fxx>::new(3);
            assert!(eng.factor(&mut S).is_ok());

            let mut M = Matrix::<$fxx>::zeros((3, 3));
            M.mul(&eng.L, &eng.L.t(), 1.0, 0.0);

            assert!(M.data().norm_inf_diff(Scopy.data()) < (1e-8 as $fxx).$tolfn());

            // now try to solve with multiple RHS
            let X = Matrix::<$fxx>::from(&[
                [1., 2.], //
                [3., 4.], //
                [5., 6.],
            ]);
            let mut B = Matrix::<$fxx>::from(&[
                [22., 32.], //
                [44., 56.], //
                [40., 52.],
            ]);

            eng.solve(&mut B);

            println!("final check {:?}", B.data.norm_inf_diff(X.data()));
            assert!(B.data.norm_inf_diff(X.data()) <= (1e-12 as $fxx).$tolfn());
        }
    };
}

generate_test_cholesky!(f32, test_cholesky_f32, sqrt);
generate_test_cholesky!(f64, test_cholesky_f64, abs);

macro_rules! generate_test_cholesky_logdet {
    ($fxx:ty, $test_name:ident, $tolfn:ident) => {
        #[test]
        #[allow(clippy::excessive_precision)]
        fn $test_name() {
            #[rustfmt::skip]
            let mut S = Matrix::<$fxx>::from(
            &[[ 8., -2., 4.],
              [-2., 12., 2.],
              [ 4.,  2., 6.]]);

            let mut eng = CholeskyEngine::<$fxx>::new(3);
            assert!(eng.factor(&mut S).is_ok());
            assert!((eng.logdet() - 5.69035945432406).abs() < (1e-10 as $fxx).$tolfn());
        }
    };
}

generate_test_cholesky_logdet!(f32, test_cholesky_logdet_f32, sqrt);
generate_test_cholesky_logdet!(f64, test_cholesky_logdet_f64, abs);
