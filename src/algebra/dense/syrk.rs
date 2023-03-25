#![allow(non_snake_case)]

use crate::algebra::{DenseMatrix, FloatT, Matrix, MatrixShape, MultiplySYRK, ShapedMatrix};

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

        let transA = A.shape().as_blas_char();
        let n = A.nrows().try_into().unwrap();
        let k = A.ncols().try_into().unwrap();
        let lda = if A.shape() == MatrixShape::N { n } else { k };
        let ldc = n;

        T::xsyrk(
            b'U',
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
    let A = Matrix::new_from_slice((m, n), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let mut AAt = Matrix::<f64>::zeros((m, m));
    AAt.syrk(&A, 1.0, 0.0);
    //NB: fills upper triangle only
    assert!(AAt.data() == [14.0, 0.0, 32.0, 77.0]);

    let mut AtA = Matrix::<f64>::zeros((n, n));
    AtA.data_mut().fill(1.0);
    AtA.syrk(&A.t(), 2.0, 1.0);
    //NB: fills upper triangle only
    println!("AtA = {}", AtA);
    assert!(AtA.data() == [35.0, 1.0, 1.0, 45.0, 59.0, 1.0, 55.0, 73.0, 91.0]);
}
