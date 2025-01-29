#![allow(non_snake_case)]

use crate::algebra::{
    Adjoint, DenseMatrix, DenseStorageMatrix, FloatT, Matrix, MatrixShape, MultiplyGEMV,
    ShapedMatrix,
};

// PJG: This needs to be over generic storage
impl<S, T> MultiplyGEMV<T> for DenseStorageMatrix<S, T>
where
    T: FloatT,
    S: AsRef<[T]>,
{
    // implements y = αA*x + βy
    fn gemv(&self, x: &[T], y: &mut [T], α: T, β: T) {
        let (m, n) = self.size();
        Matrix::<T>::_gemv(self.shape(), m, n, α, self.data(), x, β, y);
    }
}

impl<S, T> MultiplyGEMV<T> for Adjoint<'_, DenseStorageMatrix<S, T>>
where
    T: FloatT,
    S: AsRef<[T]>,
{
    // implements y = αA'*x + βy
    fn gemv(&self, x: &[T], y: &mut [T], α: T, β: T) {
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

macro_rules! generate_test_gemv {
    ($fxx:ty, $test_name:ident) => {
        #[test]
        fn $test_name() {
            use crate::algebra::Matrix;

            #[rustfmt::skip]
            let A = Matrix::<$fxx>::from(
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
    };
}

generate_test_gemv!(f32, test_gemv_f32);
generate_test_gemv!(f64, test_gemv_f64);
