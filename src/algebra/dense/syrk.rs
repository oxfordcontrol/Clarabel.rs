#![allow(non_snake_case)]

extern crate openblas_src;
use crate::algebra::{DenseMatrix, Matrix, MatrixShape, MultiplySYRK, ShapedMatrix};
use blas::{dsyrk, ssyrk};

macro_rules! impl_blas_syrk {
    ($T:ty, $SYRK:path) => {
        impl MultiplySYRK for Matrix<$T> {
            type T = $T;

            // implements self = C = αA*A' + βC
            fn syrk<MATA>(&mut self, A: &MATA, α: $T, β: $T) -> &Self
            where
                MATA: DenseMatrix<T = $T>,
            {
                assert!(self.nrows() == A.nrows());
                assert!(self.ncols() == A.nrows());

                let transA = A.shape().as_blas_char();
                let n = A.nrows().try_into().unwrap();
                let k = A.ncols().try_into().unwrap();
                let lda = if A.shape() == MatrixShape::N { n } else { k };
                let ldc = n;

                unsafe {
                    $SYRK(
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
                }
                self
            }
        }
    };
}
impl_blas_syrk!(f32, ssyrk);
impl_blas_syrk!(f64, dsyrk);

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
