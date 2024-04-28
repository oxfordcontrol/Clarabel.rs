#![allow(non_snake_case)]

use crate::algebra::*;

pub(crate) struct LuSolver {
    // BLAS workspace (allocated vecs only)
    ipiv: Vec<i32>,
}

impl LuSolver {
    pub fn new() -> Self {
        let ipiv = vec![];
        Self { ipiv }
    }
}

impl<T> SolveLU<T> for LuSolver
where
    T: FloatT,
{
    fn lusolve(
        &mut self,
        A: &mut Matrix<T>,
        B: &mut Matrix<T>,
    ) -> Result<(), DenseFactorizationError> {
        if !A.is_square() || A.ncols() != B.nrows() {
            return Err(DenseFactorizationError::IncompatibleDimension);
        }

        let n = A.nrows().try_into().unwrap();
        let nrhs = B.ncols().try_into().unwrap();
        let lda = A.ncols().try_into().unwrap();
        let a = A.data_mut();
        self.ipiv.resize(n as usize, 0);
        let ipiv = self.ipiv.as_mut_slice();
        let ldb = B.nrows().try_into().unwrap();
        let b = B.data_mut();
        let info = &mut 0_i32;

        T::xgesv(n, nrhs, a, lda, ipiv, b, ldb, info);

        if *info != 0 {
            return Err(DenseFactorizationError::LU(*info));
        }

        Ok(())
    }
}

#[test]
fn test_lu() {
    let mut A = Matrix::from(&[
        [3., 2., 4.], //
        [2., 0., 2.], //
        [4., 2., 3.], //
    ]);

    let mut B = Matrix::from(&[
        [-5., 13.], //
        [-2., 4.],  //
        [-2., 9.],  //
    ]);

    let X = Matrix::from(&[
        [1., -1.], //
        [0., 2.],  //
        [-2., 3.], //
    ]);

    let mut lu = LuSolver::new();
    lu.lusolve(&mut A, &mut B).unwrap();
    assert_eq!(B, X);
}
