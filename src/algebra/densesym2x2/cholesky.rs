#![allow(non_snake_case)]

use super::DenseMatrixSym2;
use crate::algebra::*;

impl<T> DenseMatrixSym2<T>
where
    T: FloatT,
{
    //  Unrolled 2x2 cholesky decomposition without pivoting
    //  Returns an error for a non-positive pivot and the
    //  factorization is not completed
    //
    //  NB: this is only marginally slower than the explicit
    //  3x3 LDL decomposition, which would avoid sqrts.

    pub fn cholesky_2x2_explicit_factor(
        &mut self,
        A: &DenseMatrixSym2<T>,
    ) -> Result<(), DenseFactorizationError> {
        let L = self;

        let t = A[(0, 0)];
        if t <= T::zero() {
            // positive value k means non-positive pivot leading minor k
            return Err(DenseFactorizationError::Cholesky(1));
        }

        L[(0, 0)] = t.sqrt();
        L[(1, 0)] = A[(1, 0)] / L[(0, 0)];

        let t = A[(1, 1)] - L[(1, 0)] * L[(1, 0)];

        if t <= T::zero() {
            return Err(DenseFactorizationError::Cholesky(2));
        }

        L[(1, 1)] = t.sqrt();

        Ok(())
    }

    // Unrolled 2x2 forward/backward substition for a Cholesky factor

    pub fn cholesky_2x2_explicit_solve(&self, x: &mut [T], b: &[T]) {
        let L = self;

        // Solve Lc = b
        let c0 = b[0] / L[(0, 0)];
        let c1 = (b[1] - L[(1, 0)] * c0) / L[(1, 1)];

        // Solve L^T x = c
        x[1] = c1 / L[(1, 1)];
        x[0] = (c0 - L[(1, 0)] * x[1]) / L[(0, 0)];
    }
}
