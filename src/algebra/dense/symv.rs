#![allow(non_snake_case)]

extern crate openblas_src;
use crate::algebra::{Adjoint, DenseMatrix, Matrix, MatrixTriangle, MultiplySYMV, ShapedMatrix};
use blas::{dsymv, ssymv};

macro_rules! impl_blas_symv {
    ($T:ty, $SYMV:path) => {
        impl MultiplySYMV for Matrix<$T> {
            type T = $T;
            // implements y = αA*x + βy
            fn symv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T) {
                let (m, n) = self.size();
                Matrix::<$T>::_symv(m, n, α, &self.data(), x, β, y);
            }
        }

        //PJG - NB: implementation is identical to the above, but it is
        //not obvious how to make a blanket implementation that works
        impl<'a> MultiplySYMV for Adjoint<'a, Matrix<$T>> {
            type T = $T;
            // implements y = αA*x + βy
            fn symv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T) {
                let (m, n) = self.src.size();
                Matrix::<$T>::_symv(m, n, α, &self.data(), x, β, y);
            }
        }

        impl Matrix<$T> {
            #[allow(clippy::too_many_arguments)]
            fn _symv(m: usize, n: usize, α: $T, a: &[$T], x: &[$T], β: $T, y: &mut [$T]) {
                assert!(m == n);

                // standard BLAS ?gemv arguments for computing matrix-vector product
                let uplo = MatrixTriangle::Triu.as_blas_char();
                let n = n.try_into().unwrap();
                let lda = n;
                let incx = 1;
                let incy = 1;

                unsafe {
                    $SYMV(uplo, n, α, a, lda, x, incx, β, y, incy);
                }
            }
        }
    };
}
impl_blas_symv!(f32, ssymv);
impl_blas_symv!(f64, dsymv);

#[test]
fn test_gsymv() {
    use crate::algebra::Matrix;
    let (m, n) = (3, 3);
    let a = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0];
    let A = Matrix::new_from_slice((m, n), &a);

    let x = vec![1., -2., 3.];
    let mut y = vec![-4., -1., 3.];
    A.symv(&x, &mut y, 2.0, 3.0);
    assert!(y == [6.0, 19.0, 33.0]);

    let x = vec![1., -2., 3.];
    let mut y = vec![-4., -1., 3.];
    A.t().symv(&x, &mut y, 2.0, 3.0);
    assert!(y == [6.0, 19.0, 33.0]);
}
