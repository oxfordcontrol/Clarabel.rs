#![allow(non_snake_case)]
use crate::algebra::{DenseFactorizationError, DenseMatrix, Matrix};

pub(crate) trait FactorEigen {
    type T;
    // computes eigenvalues only (full set)
    fn eigvals(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;
    // computes eigenvalues and vectors (full set)
    #[allow(dead_code)] //PJG: not currently used anywhere
    fn eigen(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;
}

pub(crate) trait FactorCholesky {
    type T;
    // computes the Cholesky decomposition.  Only the upper
    // part of the input A will be referenced. The Cholesky factor
    // is stored in self.L
    fn factor(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;

    // Solve AX = B, where B is matrix with (possibly) multiple columns.
    // Uses previously computed factor from the `factor` function.
    // B is modified in place and stores X after call.
    fn solve(&mut self, B: &mut Matrix<Self::T>);

    // computes log(det(X)) for the matrix X = LL^T
    fn logdet(&self) -> Self::T;
}

pub(crate) trait FactorSVD {
    type T;
    // compute "economy size" SVD.  Values in A are overwritten
    // as internal working space.
    fn factor(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;

    // Solve AX = B, where B is matrix with (possibly) multiple columns.
    // Uses previously computed SVD factors.   Computes a solution using
    // the pseudoinverse of A when A is rank deficient / non-square.
    // B is modified in place and stores X after call.
    fn solve(&mut self, B: &mut Matrix<Self::T>);
}

pub(crate) trait MultiplySYRK {
    type T;
    fn syrk<MATA>(&mut self, A: &MATA, α: Self::T, β: Self::T) -> &Self
    where
        MATA: DenseMatrix<T = Self::T>;
}

pub(crate) trait MultiplySYR2K {
    type T;
    fn syr2k(
        &mut self, A: &Matrix<Self::T>, B: &Matrix<Self::T>, α: Self::T, β: Self::T
    ) -> &Self;
}

pub(crate) trait MultiplyGEMM {
    type T;
    fn mul<MATA, MATB>(&mut self, A: &MATA, B: &MATB, α: Self::T, β: Self::T) -> &Self
    where
        MATB: DenseMatrix<T = Self::T>,
        MATA: DenseMatrix<T = Self::T>;
}

// Solve AX = B.  A will be corrupted post solution, and B will be
// overwritten with the solution X.
#[allow(dead_code)] //PJG: not currently used anywhere
pub(crate) trait SolveLU<T> {
    fn lusolve(
        &mut self,
        A: &mut Matrix<T>,
        B: &mut Matrix<T>,
    ) -> Result<(), DenseFactorizationError>;
}

#[allow(dead_code)] //PJG: not currently used anywhere
pub(crate) trait MultiplyGEMV {
    type T;
    fn gemv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T);
}

#[allow(dead_code)] //PJG: not currently used anywhere
pub(crate) trait MultiplySYMV {
    type T;
    fn symv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T);
}
