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

#[cfg(test)]
mod test {

    use super::*;
    use crate::algebra::*;

    #[test]
    fn test_cholesky_2x2() {
        let As = DenseMatrixSym2 {
            data: [4.0, -2.0, 6.0],
        };

        let xtrue = vec![1.0, 2.0];
        let mut b = vec![0.0; 3];
        As.mul(&mut b, &xtrue);

        let mut L = DenseMatrixSym2::zeros();
        L.cholesky_2x2_explicit_factor(&As).unwrap();

        // reconstuct M = LL^T
        let mut Lfull: Matrix<f64> = L.clone().into();

        // PJG: L is not actually symmetric, but is rather
        // the cholesky factor packed into a a triangle.
        // Zero out the upper triangle explicitly.   Probably
        // a tril function would be better, or some
        // TriangularMatrix type wrapper.
        Lfull[(0, 1)] = 0.0;

        let mut M = Matrix::zeros((2, 2));
        M.mul(&Lfull, &Lfull.t(), 1.0, 0.0);

        // solve
        let mut xsolve = vec![0.0; 2];
        L.cholesky_2x2_explicit_solve(&mut xsolve, &b);

        assert!(xsolve.norm_inf_diff(&xtrue) < 1e-10);
    }
}
