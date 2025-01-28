#![allow(non_snake_case)]

use crate::algebra::*;

impl<T> MultiplySYRK<T> for Matrix<T>
where
    T: FloatT,
{
    // implements self = C = αA*A' + βC
    // The matrix input A can itself be
    // an Adjoint matrix, in which case the
    // result amounts to C = αA'*A + βC
    fn syrk<MATA>(&mut self, A: &MATA, α: T, β: T)
    where
        MATA: DenseMatrix<T>,
    {
        assert!(self.nrows() == A.nrows());
        assert!(self.ncols() == A.nrows());

        if self.nrows() == 0 {
            return;
        }

        let uplo = MatrixTriangle::Triu.as_blas_char();
        let transA = A.shape().as_blas_char();
        let n = A.nrows().try_into().unwrap();
        let k = A.ncols().try_into().unwrap();
        let lda = if A.shape() == MatrixShape::N { n } else { k };
        let ldc = n;

        #[rustfmt::skip]
        T::xsyrk(uplo,transA,n,k,α,A.data(),lda,β,self.data_mut(),ldc);
    }
}

macro_rules! generate_test_syrk {
    ($fxx:ty, $test_name:ident) => {
        #[test]
        fn $test_name() {
            let (m, n) = (2, 3);
            let A = Matrix::<$fxx>::from(&[
                [1., 2., 3.], //
                [4., 5., 6.], //
            ]);

            let mut AAt = Matrix::<$fxx>::zeros((m, m));
            AAt.syrk(&A, 1.0, 0.0);

            //NB: writes to upper triangle only
            let AAt_test = Matrix::<$fxx>::from(&[
                [14., 32.], //
                [0., 77.],  //
            ]);

            assert_eq!(AAt, AAt_test);

            let mut AtA = Matrix::<$fxx>::zeros((n, n));
            AtA.data_mut().fill(1.0);
            AtA.syrk(&A.t(), 2.0, 1.0);

            //NB: writes to upper triangle only
            let AtA_test = Matrix::<$fxx>::from(&[
                [35., 45., 55.], //
                [1., 59., 73.],  //
                [1., 1., 91.],   //
            ]);

            assert_eq!(AtA, AtA_test);
        }
    };
}

generate_test_syrk!(f32, test_syrk_f32);
generate_test_syrk!(f64, test_syrk_f64);
