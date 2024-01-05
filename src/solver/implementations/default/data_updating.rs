#![allow(non_snake_case)]
use crate::algebra::*;

use super::DefaultSolver;

pub trait MatrixProblemDataUpdate<T: FloatT> {
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
    ) -> Result<(), SparseFormatError>;
}

impl<T> MatrixProblemDataUpdate<T> for CscMatrix<T>
where
    T: FloatT,
{
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
    ) -> Result<(), SparseFormatError> {
        self.check_equal_sparsity(M)?;
        let v = &self.nzval;
        v.update_matrix(M, lscale, rscale)
    }
}

impl<T, Data> MatrixProblemDataUpdate<T> for Data
where
    T: FloatT,
    Data: AsRef<[T]>,
{
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
    ) -> Result<(), SparseFormatError> {
        let data = self.as_ref();
        if data.is_empty() {
            return Ok(());
        }

        if data.len() != M.nzval.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        M.nzval.copy_from_slice(data);

        // reapply original equilibration
        M.lrscale(lscale, rscale);

        Ok(())
    }
}

pub trait VectorProblemDataUpdate<T: FloatT> {
    fn update_vector(&self, v: &mut [T], scale: &[T]) -> Result<(), SparseFormatError>;
}

impl<T, Data> VectorProblemDataUpdate<T> for Data
where
    T: FloatT,
    Data: AsRef<[T]>,
{
    fn update_vector(&self, v: &mut [T], scale: &[T]) -> Result<(), SparseFormatError> {
        let data = self.as_ref();
        if data.is_empty() {
            return Ok(());
        }

        if data.len() != v.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        v.copy_from_slice(data);

        //reapply original equilibration
        v.hadamard(scale);

        Ok(())
    }
}

impl<T> DefaultSolver<T>
where
    T: FloatT,
{
    /// Overwrites internal problem data structures in a solver object with new data, avoiding new memory allocations.   
    /// See `update_P``, `update_q`, `update_A`, `update_b` for allowable inputs.

    pub fn update_data<
        DataP: MatrixProblemDataUpdate<T>,
        Dataq: VectorProblemDataUpdate<T>,
        DataA: MatrixProblemDataUpdate<T>,
        Datab: VectorProblemDataUpdate<T>,
    >(
        &mut self,
        P: &DataP,
        q: &Dataq,
        A: &DataA,
        b: &Datab,
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
    pub fn update_P<Data: MatrixProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        data.update_matrix(
            &mut self.data.P,
            &self.data.equilibration.d,
            &self.data.equilibration.d,
        )?;
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
    pub fn update_A<Data: MatrixProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        data.update_matrix(
            &mut self.data.A,
            &self.data.equilibration.e,
            &self.data.equilibration.d,
        )?;
        // overwrite KKT data
        self.kktsystem.update_A(&self.data.A);
        Ok(())
    }

    /// Overwrites the `q` vector data in an existing solver object.  No action is taken if the input is empty.
    ///
    //PJG: Error type is not ideal here.   Maybe need a generic user input
    //error type, which can conatin a SparseFormatError or a PresolveDisabled error
    pub fn update_q<Data: VectorProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        let d = &self.data.equilibration.d;
        let dinv = &self.data.equilibration.dinv;
        data.update_vector(&mut self.data.q, d)?;

        //recover unscaled norm
        self.data.normq = self.data.q.norm_inf_scaled(dinv);

        Ok(())
    }

    /// Overwrites the `b` vector data in an existing solver object.  No action is taken if the input is empty.
    pub fn update_b<Data: VectorProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), SparseFormatError> {
        self.check_presolve_disabled()?;
        let e = &self.data.equilibration.e;
        let einv = &self.data.equilibration.einv;
        data.update_vector(&mut self.data.b, e)?;

        //recover unscaled norm
        self.data.normb = self.data.b.norm_inf_scaled(einv);

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
