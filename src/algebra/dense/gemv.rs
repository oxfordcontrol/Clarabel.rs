#![allow(non_snake_case)]

use crate::algebra::{
    Adjoint, DenseMatrix, FloatT, Matrix, MatrixShape, MultiplyGEMV, ShapedMatrix,
};

impl<T> MultiplyGEMV for Matrix<T>
where
    T: FloatT,
{
    type T = T;
    // implements y = αA*x + βy
    fn gemv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T) {
        let (m, n) = self.size();
        Matrix::<T>::_gemv(self.shape(), m, n, α, self.data(), x, β, y);
    }
}

impl<'a, T> MultiplyGEMV for Adjoint<'a, Matrix<T>>
where
    T: FloatT,
{
    type T = T;
    // implements y = αA'*x + βy
    fn gemv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T) {
        let (m, n) = self.src.size(); //NB: size of A, not A'
        Matrix::<T>::_gemv(self.shape(), m, n, α, self.data(), x, β, y);
    }
}

impl<T> Matrix<T>
where
    T: FloatT,
{
    #[allow(clippy::too_many_arguments)]
    fn _gemv(trans: MatrixShape, m: usize, n: usize, α: T, a: &[T], x: &[T], β: T, y: &mut [T]) {
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

        T::xgemv(trans, m, n, α, a, lda, x, incx, β, y, incy);
    }
}

#[test]
fn test_gemv() {
    use crate::algebra::Matrix;

    #[rustfmt::skip]
    let A = Matrix::from(
        &[[ 1.,  2., 3.],
          [ 4.,  5., 6.]]);

    let x = vec![1., 2., 3.];
    let mut y = vec![-1., -2.];
    A.gemv(&x, &mut y, 2.0, 3.0);
    assert!(y == [25.0, 58.0]);

    let x = vec![1., 2.];
    let mut y = vec![-1., -2., -3.];
    A.t().gemv(&x, &mut y, 2.0, 3.0);
    assert!(y == [15.0, 18.0, 21.0]);
}
