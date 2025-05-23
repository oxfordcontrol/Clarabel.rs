#![allow(non_snake_case)]

use super::DenseMatrixSym3;
use crate::algebra::*;

impl<T> DenseMatrixSym3<T>
where
    T: FloatT,
{
    //  Unrolled 3x3 cholesky decomposition without pivoting
    //  Returns an error for a non-positive pivot and the
    //  factorization is not completed
    //
    //  NB: this is only marginally slower than the explicit
    //  3x3 LDL decomposition, which would avoid sqrts.

    pub(crate) fn cholesky_3x3_explicit_factor(
        &mut self,
        A: &DenseMatrixSym3<T>,
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
        L[(2, 0)] = A[(2, 0)] / L[(0, 0)];
        L[(2, 1)] = (A[(2, 1)] - L[(1, 0)] * L[(2, 0)]) / L[(1, 1)];

        let t = A[(2, 2)] - L[(2, 0)] * L[(2, 0)] - L[(2, 1)] * L[(2, 1)];

        if t <= T::zero() {
            return Err(DenseFactorizationError::Cholesky(3));
        }
        L[(2, 2)] = t.sqrt();

        Ok(())
    }

    // Unrolled 3x3 forward/backward substition for a Cholesky factor

    pub(crate) fn cholesky_3x3_explicit_solve(&self, x: &mut [T], b: &[T]) {
        let L = self;

        // Forward substitution: Solve Lc = b
        let c0 = b[0] / L[(0, 0)];
        let c1 = (b[1] - L[(1, 0)] * c0) / L[(1, 1)];
        let c2 = (b[2] - L[(2, 0)] * c0 - L[(2, 1)] * c1) / L[(2, 2)];

        // Backward substitution: Solve L^T x = c
        x[2] = c2 / L[(2, 2)];
        x[1] = (c1 - L[(2, 1)] * x[2]) / L[(1, 1)];
        x[0] = (c0 - L[(1, 0)] * x[1] - L[(2, 0)] * x[2]) / L[(0, 0)];
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::algebra::*;

    #[test]
    fn test_cholesky_3x3() {
        let As = DenseMatrixSym3 {
            data: [4.0, -2.0, 6.0, 1.0, 2.0, 9.0],
        };

        let xtrue = vec![1.0, 2.0, 3.0];
        let mut b = vec![0.0; 3];
        As.mul(&mut b, &xtrue);

        let mut L = DenseMatrixSym3::zeros();
        L.cholesky_3x3_explicit_factor(&As).unwrap();

        // reconstuct M = LL^T
        let mut Lfull: Matrix<f64> = L.clone().into();

        // PJG: L is not actually symmetric, but is rather
        // the cholesky factor packed into a a triangle.
        // Zero out the upper triangle explicitly.   Probably
        // a tril function would be better, or some
        // TriangularMatrix type wrapper.
        Lfull[(0, 1)] = 0.0;
        Lfull[(0, 2)] = 0.0;
        Lfull[(1, 2)] = 0.0;

        // solve
        let mut xsolve = vec![0.0; 3];
        L.cholesky_3x3_explicit_solve(&mut xsolve, &b);

        assert!(xsolve.norm_inf_diff(&xtrue) < 1e-10);
    }
}
