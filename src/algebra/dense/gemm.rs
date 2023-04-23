#![allow(non_snake_case)]

use crate::algebra::{DenseMatrix, FloatT, Matrix, MatrixShape, MultiplyGEMM, ShapedMatrix};

impl<T> MultiplyGEMM for Matrix<T>
where
    T: FloatT,
{
    type T = T;
    // implements self = C = αA*B + βC
    fn mul<MATA, MATB>(&mut self, A: &MATA, B: &MATB, α: T, β: T) -> &Self
    where
        MATA: DenseMatrix<T = T>,
        MATB: DenseMatrix<T = T>,
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

        T::xgemm(
            transA,
            transB,
            m,
            n,
            k,
            α,
            A.data(),
            lda,
            B.data(),
            ldb,
            β,
            self.data_mut(),
            ldc,
        );
        self
    }
}

#[test]
fn test_gemm() {
    let (m, n, k) = (2, 4, 3);
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = vec![
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

    let mut A = Matrix::zeros((m, k));
    let mut B = Matrix::zeros((k, n));
    let mut C = Matrix::<f64>::zeros((m, n));
    A.copy_from_slice(&a);
    B.copy_from_slice(&b);
    C.copy_from_slice(&c);
    C.mul(&A, &B, 1.0, 1.0);

    assert!(C.data() == vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0]);

    // new from slice and transposed multiply
    let A = Matrix::new_from_slice((m, k), &a);
    let B = Matrix::new_from_slice((k, n), &b);
    let mut C = Matrix::<f64>::zeros((n, m));
    C.mul(&B.t(), &A.t(), 1.0, 0.0);

    assert!(C.data() == vec![38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0]);
}
