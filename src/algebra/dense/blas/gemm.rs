#![allow(non_snake_case)]

use crate::algebra::*;

impl<S, T> MultiplyGEMM<T> for DenseStorageMatrix<S, T>
where
    T: FloatT,
    S: AsRef<[T]> + AsMut<[T]>,
{
    // implements self = C = αA*B + βC
    fn mul<MATA, MATB>(&mut self, A: &MATA, B: &MATB, α: T, β: T) -> &Self
    where
        MATA: DenseMatrix<T>,
        MATB: DenseMatrix<T>,
    {
        assert!(A.ncols() == B.nrows() && self.nrows() == A.nrows() && self.ncols() == B.ncols());

        if self.nrows() == 0 || self.ncols() == 0 {
            return self;
        }

        // standard BLAS ?gemm arguments for computing
        // general matrix-matrix multiply
        let transA = A.shape().as_blas_char();
        let transB = B.shape().as_blas_char();
        let m = A.nrows().try_into().unwrap();
        let n = B.ncols().try_into().unwrap();
        let k = A.ncols().try_into().unwrap();
        let lda = if A.shape() == MatrixShape::N { m } else { k };
        let ldb = if B.shape() == MatrixShape::N { k } else { n };
        let ldc = m;

        #[rustfmt::skip]
        T::xgemm(transA,transB,m,n,k,α,A.data(),lda,B.data(),ldb,β,self.data_mut(),ldc);

        self
    }
}

macro_rules! generate_test_gemm {
    ($fxx:ty, $test_name:ident) => {
        #[test]
        fn $test_name() {
            let (m, n, k) = (2, 4, 3);
            let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
            let b = vec![
                1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
            ];
            let c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

            let mut A = Matrix::<$fxx>::zeros((m, k));
            let mut B = Matrix::<$fxx>::zeros((k, n));
            let mut C = Matrix::<$fxx>::zeros((m, n));
            A.copy_from_slice(&a);
            B.copy_from_slice(&b);
            C.copy_from_slice(&c);
            C.mul(&A, &B, 1.0, 1.0);

            assert!(C.data() == vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);

            // new from slice and transposed multiply
            let A = Matrix::<$fxx>::new_from_slice((m, k), &a);
            let B = Matrix::<$fxx>::new_from_slice((k, n), &b);
            let mut C = Matrix::<$fxx>::zeros((n, m));
            C.mul(&B.t(), &A.t(), 1.0, 0.0);

            assert!(C.data() == vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
        }
    };
}

generate_test_gemm!(f32, test_gemm_f32);
generate_test_gemm!(f64, test_gemm_f64);
