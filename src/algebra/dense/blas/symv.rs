#![allow(non_snake_case)]

use crate::algebra::{
    DenseMatrix, FloatT, Matrix, MatrixTriangle, MultiplySYMV, ShapedMatrix, Symmetric,
};

impl<T> MultiplySYMV for Symmetric<'_, Matrix<T>>
where
    T: FloatT,
{
    type T = T;
    // implements y = αA*x + βy
    fn symv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T) {
        let (m, n) = self.size();
        assert!(m == n);

        // standard BLAS ?symv arguments for computing matrix-vector product
        let uplo = MatrixTriangle::Triu.as_blas_char();
        let n = n.try_into().unwrap();
        let a = self.src.data();
        let lda = n;
        let incx = 1;
        let incy = 1;
        T::xsymv(uplo, n, α, a, lda, x, incx, β, y, incy);
    }
}

macro_rules! generate_test_gsymv {
    ($fxx:ty, $test_name:ident) => {
        #[test]
        fn $test_name() {
            #[rustfmt::skip]
            let A = Matrix::<$fxx>::from(&[
                [ 1.,  2.,   4.], 
                [ 0.,  3.,   5.], 
                [ 0.,  0.,   6.],
            ]);

            let x = vec![1., -2., 3.];
            let mut y = vec![-4., -1., 3.];
            A.sym().symv(&x, &mut y, 2.0, 3.0);
            assert_eq!(y, [6.0, 19.0, 33.0]);
        } 
    };
}

generate_test_gsymv!(f32, test_gsymv_f32);
generate_test_gsymv!(f64, test_gsymv_f64);
