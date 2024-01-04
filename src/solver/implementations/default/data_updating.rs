#![allow(non_snake_case)]
use crate::algebra::*;
use crate::solver::{scale_A, scale_P, scale_b, scale_q};
use std::convert::From;

use super::DefaultSolver;

// Enum type allowing for flexible user input of matrix data updates.

pub enum MatrixUpdateDataSource<'a, T: FloatT> {
    CscMatrix(&'a CscMatrix<T>),
    Slice(&'a [T]),
}

impl<'a, T> From<&'a [T]> for MatrixUpdateDataSource<'a, T>
where
    T: FloatT,
{
    fn from(v: &'a [T]) -> Self {
        MatrixUpdateDataSource::Slice(v)
    }
}

impl<'a, T> From<&'a Vec<T>> for MatrixUpdateDataSource<'a, T>
where
    T: FloatT,
{
    fn from(v: &'a Vec<T>) -> Self {
        MatrixUpdateDataSource::Slice(v)
    }
}

impl<'a, T> From<&'a [T; 0]> for MatrixUpdateDataSource<'a, T>
where
    T: FloatT,
{
    fn from(v: &'a [T; 0]) -> Self {
        MatrixUpdateDataSource::Slice(v)
    }
}

impl<'a, T> From<&'a CscMatrix<T>> for MatrixUpdateDataSource<'a, T>
where
    T: FloatT,
{
    fn from(v: &'a CscMatrix<T>) -> Self {
        MatrixUpdateDataSource::CscMatrix(v)
    }
}

impl<T> DefaultSolver<T>
where
    T: FloatT,
{
    /// Overwrites internal problem data structures in a solver object with new data, avoiding new memory allocations.   
    /// See `update_P``, `update_q`, `update_A`, `update_b` for allowable inputs.

    pub fn update_data<
        'a,
        CscOrSliceP: Into<MatrixUpdateDataSource<'a, T>>,
        CscOrSliceA: Into<MatrixUpdateDataSource<'a, T>>,
    >(
        &mut self,
        P: CscOrSliceP,
        q: &[T],
        A: CscOrSliceA,
        b: &[T],
    ) -> Result<(), SparseFormatError> {
        self.update_P(P)?;
        self.update_q(q)?;
        self.update_A(A)?;
        self.update_b(b)?;

        Ok(())
    }

    /// Overwrites the `P` matrix data in an existing solver object.   The input `P` can be    
    ///
    /// - a nonempty Vector, in which case the nonzero values of the original `P` are overwritten, preserving the sparsity pattern, or
    ///
    /// - a SparseMatrixCSC, in which case the input must match the sparsity pattern of the upper triangular part of the original `P`.   
    ///
    /// - an empty vector, in which case no action is taken.
    ///
    pub fn update_P<'a, CscOrSlice: Into<MatrixUpdateDataSource<'a, T>>>(
        &mut self,
        data: CscOrSlice,
    ) -> Result<(), SparseFormatError> {
        let data = data.into();
        match data {
            MatrixUpdateDataSource::CscMatrix(P) => {
                P.check_equal_sparsity(&self.data.P)?;
                self.update_P_slice(&P.nzval)
            }
            MatrixUpdateDataSource::Slice(v) => self.update_P_slice(v),
        }
    }

    fn update_P_slice(&mut self, v: &[T]) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        if v.is_empty() {
            return Ok(());
        }

        if v.len() != self.data.P.nzval.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        self.data.P.nzval.copy_from_slice(v);

        // reapply original equilibration
        scale_P(&mut self.data.P, &self.data.equilibration.d);

        // overwrite KKT data
        self.kktsystem.update_P(&self.data.P);
        Ok(())
    }

    /// Overwrites the `A` matrix data in an existing solver object.   The input `A` can be    
    ///
    /// - a nonempty Vector, in which case the nonzero values of the original `A` are overwritten, preserving the sparsity pattern, or
    ///
    /// - a SparseMatrixCSC, in which case the input must match the sparsity pattern of the original `A`.   
    ///
    /// - an empty vector, in which case no action is taken.
    ///
    pub fn update_A<'a, CscOrVec: Into<MatrixUpdateDataSource<'a, T>>>(
        &mut self,
        data: CscOrVec,
    ) -> Result<(), SparseFormatError> {
        let data = data.into();
        match data {
            MatrixUpdateDataSource::CscMatrix(A) => {
                A.check_equal_sparsity(&self.data.A)?;
                self.update_A_slice(&A.nzval)
            }
            MatrixUpdateDataSource::Slice(v) => self.update_A_slice(v),
        }
    }

    fn update_A_slice(&mut self, v: &[T]) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        if v.is_empty() {
            return Ok(());
        }

        if v.len() != self.data.A.nzval.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        self.data.A.nzval.copy_from_slice(v);

        // reapply original equilibration
        scale_A(
            &mut self.data.A,
            &self.data.equilibration.e,
            &self.data.equilibration.d,
        );

        // overwrite KKT data
        self.kktsystem.update_A(&self.data.A);

        Ok(())
    }

    /// Overwrites the `q` vector data in an existing solver object.  No action is taken if the input is empty.
    ///
    //PJG: Error type is not ideal here.   Maybe need a generic user input
    //error type, which can conatin a SparseFormatError or a PresolveDisabled error
    pub fn update_q(&mut self, q: &[T]) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        if q.is_empty() {
            return Ok(());
        }

        if q.len() != self.data.q.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        self.data.q.copy_from_slice(q);

        //recompute unscaled norm
        self.data.normq = self.data.q.norm_inf();

        //reapply original equilibration
        scale_q(&mut self.data.q, &self.data.equilibration.d);

        Ok(())
    }

    /// Overwrites the `b` vector data in an existing solver object.  No action is taken if the input is empty.
    pub fn update_b(&mut self, b: &[T]) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        if b.is_empty() {
            return Ok(());
        }

        if b.len() != self.data.b.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        self.data.b.copy_from_slice(b);

        //recompute unscaled norm
        self.data.normb = self.data.b.norm_inf();

        //reapply original equilibration
        scale_b(&mut self.data.b, &self.data.equilibration.e);

        Ok(())
    }

    //PJG: This error type is incorrect.  Used in multiple places in this file
    fn check_presolve_disabled(&self) -> Result<(), SparseFormatError> {
        if self.settings.presolve_enable {
            Err(SparseFormatError::IncompatibleDimension)
        } else {
            Ok(())
        }
    }
}
