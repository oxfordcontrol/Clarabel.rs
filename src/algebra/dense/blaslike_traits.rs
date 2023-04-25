#![allow(non_snake_case)]
use crate::algebra::{DenseFactorizationError, DenseMatrix, Matrix};

pub(crate) trait FactorEigen {
    type T;
    // computes eigenvalues only (full set)
    fn eigvals(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;
    // computes eigenvalues and vectors (full set)
    fn eigen(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;
}

pub(crate) trait FactorCholesky {
    type T;
    // computes the Cholesky decomposition.  Only the upper
    // part of the input A will be referenced, and A will
    // by modified in place.   The Cholesky factor is stored
    // in self.L
    fn cholesky(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;
}

pub(crate) trait FactorSVD {
    type T;
    // compute "economy size" SVD.  Values in A are overwritten
    // as internal working space.
    fn svd(&mut self, A: &mut Matrix<Self::T>) -> Result<(), DenseFactorizationError>;
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

pub(crate) trait MultiplyGEMV {
    type T;
    fn gemv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T);
}

pub(crate) trait MultiplySYMV {
    type T;
    fn symv(&self, x: &[Self::T], y: &mut [Self::T], α: Self::T, β: Self::T);
}
