#![allow(non_snake_case)]

extern crate openblas_src;
use crate::algebra::{Adjoint, DenseMatrix, Matrix, MatrixShape, MultiplyGEMV, ShapedMatrix};
use blas::{dgemv, sgemv};

macro_rules! impl_blas_gemv {
    ($T:ty, $GEMV:path) => {
        impl MultiplyGEMV for Matrix<$T> {
            type T = $T;
            // implements y = αA*x + βy
            fn gemv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T) {
                let (m, n) = self.size();
                Matrix::<$T>::_gemv(self.shape(), m, n, α, &self.data(), x, β, y);
            }
        }

        impl<'a> MultiplyGEMV for Adjoint<'a, Matrix<$T>> {
            type T = $T;
            // implements y = αA'*x + βy
            fn gemv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T) {
                let (m, n) = self.src.size(); //NB: size of A, not A'
                Matrix::<$T>::_gemv(self.shape(), m, n, α, &self.data(), x, β, y);
            }
        }

        impl Matrix<$T> {
            #[allow(clippy::too_many_arguments)]
            fn _gemv(
                trans: MatrixShape,
                m: usize,
                n: usize,
                α: $T,
                a: &[$T],
                x: &[$T],
                β: $T,
                y: &mut [$T],
            ) {
                match trans {
                    MatrixShape::N => {
                        assert!(n == x.len() && m == y.len());
                    }
                    MatrixShape::T => {
                        assert!(m == x.len() && n == y.len());
                    }
                }

                // standard BLAS ?gemv arguments for computing matrix-vector product
                let trans = trans.as_blas_char();
                let m = m.try_into().unwrap();
                let n = n.try_into().unwrap();
                let lda = m;
                let incx = 1;
                let incy = 1;

                unsafe {
                    $GEMV(trans, m, n, α, a, lda, x, incx, β, y, incy);
                }
            }
        }
    };
}
impl_blas_gemv!(f32, sgemv);
impl_blas_gemv!(f64, dgemv);

#[test]
fn test_gemv() {
    use crate::algebra::Matrix;
    let (m, n) = (2, 3);
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let A = Matrix::new_from_slice((m, n), &a);

    let x = vec![1., 2., 3.];
    let mut y = vec![-1., -2.];
    A.gemv(&x, &mut y, 2.0, 3.0);
    assert!(y == [25.0, 58.0]);

    let x = vec![1., 2.];
    let mut y = vec![-1., -2., -3.];
    A.t().gemv(&x, &mut y, 2.0, 3.0);
    assert!(y == [15.0, 18.0, 21.0]);
}
