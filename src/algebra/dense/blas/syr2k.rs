#![allow(non_snake_case)]

use crate::algebra::*;

impl<S, T> MultiplySYR2K<T> for DenseStorageMatrix<S, T>
where
    T: FloatT,
    S: AsMut<[T]> + AsRef<[T]>,
{
    // implements self = C = α(A*B' + B*A') + βC
    fn syr2k<S1, S2>(
        &mut self,
        A: &DenseStorageMatrix<S1, T>,
        B: &DenseStorageMatrix<S2, T>,
        α: T,
        β: T,
    ) where
        S1: AsRef<[T]>,
        S2: AsRef<[T]>,
    {
        assert!(self.nrows() == A.nrows());
        assert!(self.nrows() == B.nrows());
        assert!(self.ncols() == B.nrows());
        assert!(A.ncols() == B.ncols());

        if self.nrows() == 0 {
            return;
        }

        let n = self.nrows().try_into().unwrap();
        let k = A.ncols().try_into().unwrap();
        let a = A.data();
        let lda = n;
        let b = B.data();
        let ldb = n;
        let c = self.data_mut();
        let ldc = n;

        T::xsyr2k(
            MatrixTriangle::Triu.as_blas_char(),
            MatrixShape::N.as_blas_char(),
            n,
            k,
            α,
            a,
            lda,
            b,
            ldb,
            β,
            c,
            ldc,
        );
    }
}

#[test]
#[rustfmt::skip]
fn test_syr2k() {

    let A = Matrix::from(&[
        [ 1., -5.], 
        [-4.,  3.], 
        [ 2.,  6.],
    ]);

    let B = Matrix::from(&[
        [ 4.,  5.], 
        [ 2., -2.], 
        [-3., -2.],
    ]);

    let mut C = Matrix::<f64>::identity(3);

    //NB: modifies upper triangle only
    C.syr2k(&A, &B, 2., 1.);

    let Ctest = Matrix::from(&[
        [-83.,  22.,  90.], 
        [  0., -55.,  -4.], 
        [  0.,   0., -71.],
    ]);

    assert_eq!(C,Ctest);
}
