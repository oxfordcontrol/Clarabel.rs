#![allow(non_snake_case)]

use crate::algebra::{
    DenseMatrix, FloatT, Matrix, MatrixShape, MatrixTriangle, MultiplySYRK, ShapedMatrix,
};

impl<T> MultiplySYRK for Matrix<T>
where
    T: FloatT,
{
    type T = T;

    // implements self = C = αA*A' + βC
    fn syrk<MATA>(&mut self, A: &MATA, α: T, β: T) -> &Self
    where
        MATA: DenseMatrix<T = T>,
    {
        assert!(self.nrows() == A.nrows());
        assert!(self.ncols() == A.nrows());

        if self.nrows() == 0 {
            return self;
        }

        let uplo = MatrixTriangle::Triu.as_blas_char();
        let transA = A.shape().as_blas_char();
        let n = A.nrows().try_into().unwrap();
        let k = A.ncols().try_into().unwrap();
        let lda = if A.shape() == MatrixShape::N { n } else { k };
        let ldc = n;

        T::xsyrk(
            uplo,
            transA,
            n,
            k,
            α,
            A.data(),
            lda,
            β,
            self.data_mut(),
            ldc,
        );
        self
    }
}

#[test]
fn test_syrk() {
    let (m, n) = (2, 3);
    let A = Matrix::from(&[
        [1., 2., 3.], //
        [4., 5., 6.], //
    ]);

    let mut AAt = Matrix::<f64>::zeros((m, m));
    AAt.syrk(&A, 1.0, 0.0);

    //NB: writes to upper triangle only
    let AAt_test = Matrix::from(&[
        [14., 32.], //
        [0., 77.],  //
    ]);

    assert_eq!(AAt, AAt_test);

    let mut AtA = Matrix::<f64>::zeros((n, n));
    AtA.data_mut().fill(1.0);
    AtA.syrk(&A.t(), 2.0, 1.0);

    //NB: writes to upper triangle only
    let AtA_test = Matrix::from(&[
        [35., 45., 55.], //
        [1., 59., 73.],  //
        [1., 1., 91.],   //
    ]);

    assert_eq!(AtA, AtA_test);
}
