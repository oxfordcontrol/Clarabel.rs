#![allow(non_snake_case)]
use crate::algebra::*;

pub(crate) trait FactorEigen<T> {
    // computes eigenvalues only (full set)
    fn eigvals<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>;
    // computes eigenvalues and vectors (full set)
    #[allow(dead_code)] //PJG: not currently used anywhere
    fn eigen<S>(&mut self, A: &mut DenseStorageMatrix<S, T>) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>;
}

pub(crate) trait FactorCholesky<T> {
    // computes the Cholesky decomposition.  Only the upper
    // part of the input A will be referenced. The Cholesky factor
    // is stored in self.L
    fn factor<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>;

    // Solve AX = B, where B is matrix with (possibly) multiple columns.
    // Uses previously computed factor from the `factor` function.
    // B is modified in place and stores X after call.
    fn solve<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>;

    // computes log(det(X)) for the matrix X = LL^T
    fn logdet(&self) -> T;
}

pub(crate) trait FactorSVD<T> {
    // compute "economy size" SVD.  Values in A are overwritten
    // as internal working space.
    fn factor<S>(
        &mut self,
        A: &mut DenseStorageMatrix<S, T>,
    ) -> Result<(), DenseFactorizationError>
    where
        S: AsMut<[T]> + AsRef<[T]>;

    // Solve AX = B, where B is matrix with (possibly) multiple columns.
    // Uses previously computed SVD factors.   Computes a solution using
    // the pseudoinverse of A when A is rank deficient / non-square.
    // B is modified in place and stores X after call.
    fn solve<S>(&mut self, B: &mut DenseStorageMatrix<S, T>)
    where
        S: AsMut<[T]> + AsRef<[T]>;
}

pub(crate) trait MultiplySYRK<T> {
    fn syrk<MATA>(&mut self, A: &MATA, α: T, β: T)
    where
        MATA: DenseMatrix<T>;
}

pub(crate) trait MultiplySYR2K<T> {
    fn syr2k<S1, S2>(
        &mut self,
        A: &DenseStorageMatrix<S1, T>,
        B: &DenseStorageMatrix<S2, T>,
        α: T,
        β: T,
    ) where
        S1: AsRef<[T]>,
        S2: AsRef<[T]>;
}

pub(crate) trait MultiplyGEMM<T> {
    type T;
    fn mul<MATA, MATB>(&mut self, A: &MATA, B: &MATB, α: Self::T, β: Self::T) -> &Self
    where
        MATB: DenseMatrix<T>,
        MATA: DenseMatrix<T>;
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
