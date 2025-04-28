#![allow(non_snake_case)]

use super::DenseMatrixSym3;
use crate::algebra::*;

impl<T> DenseMatrixSym3<T>
where
    T: FloatT,
{
    //  Unrolled 3x3 cholesky decomposition without pivoting
    //  Returns `false` for a non-positive pivot and the
    //  factorization is not completed
    //
    //  NB: this is only marginally slower than the explicit
    //  3x3 LDL decomposition, which would avoid sqrts.

    pub fn cholesky_3x3_explicit_factor(&mut self, A: &DenseMatrixSym3<T>) -> bool {
        // PJG: This should return Result

        let L = self;

        let t = A[(0, 0)];
        if t <= T::zero() {
            return false;
        }

        L[(0, 0)] = t.sqrt();
        L[(1, 0)] = A[(1, 0)] / L[(0, 0)];

        let t = A[(1, 1)] - L[(1, 0)] * L[(1, 0)];

        if t <= T::zero() {
            return false;
        }

        L[(1, 1)] = t.sqrt();
        L[(2, 0)] = A[(2, 0)] / L[(0, 0)];
        L[(2, 1)] = (A[(2, 1)] - L[(1, 0)] * L[(2, 0)]) / L[(1, 1)];

        let t = A[(2, 2)] - L[(2, 0)] * L[(2, 0)] - L[(2, 1)] * L[(2, 1)];

        if t <= T::zero() {
            return false;
        }
        L[(2, 2)] = t.sqrt();

        true
    }

    // Unrolled 3x3 forward/backward substition for a Cholesky factor

    pub fn cholesky_3x3_explicit_solve(&self, x: &mut [T], b: &[T]) {
        let L = self;

        let c1 = b[0] / L[(0, 0)];
        let c2 = (b[1] * L[(0, 0)] - b[0] * L[(1, 0)]) / (L[(0, 0)] * L[(1, 1)]);
        let c3 = (b[2] * L[(0, 0)] * L[(1, 1)] - b[1] * L[(0, 0)] * L[(2, 1)]
            + b[0] * L[(1, 0)] * L[(2, 1)]
            - b[0] * L[(1, 1)] * L[(2, 0)])
            / (L[(0, 0)] * L[(1, 1)] * L[(2, 2)]);

        x[0] = (c1 * L[(1, 1)] * L[(2, 2)] - c2 * L[(1, 0)] * L[(2, 2)]
            + c3 * L[(1, 0)] * L[(2, 1)]
            - c3 * L[(1, 1)] * L[(2, 0)])
            / (L[(0, 0)] * L[(1, 1)] * L[(2, 2)]);
        x[1] = (c2 * L[(2, 2)] - c3 * L[(2, 1)]) / (L[(1, 1)] * L[(2, 2)]);
        x[2] = c3 / L[(2, 2)];
    }
}
