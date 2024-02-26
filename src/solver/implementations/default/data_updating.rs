#![allow(non_snake_case)]
use super::DefaultSolver;
use crate::algebra::*;
use core::iter::Zip;
use core::slice::Iter;
use thiserror::Error;

/// Error type returned by user data update utilities, e.g. [`check_format`](crate::algebra::CscMatrix::check_format) utility.
#[derive(Error, Debug)]
pub enum DataUpdateError {
    #[error("Data updates are not allowed when presolve is enabled")]
    PresolveEnabled,
    #[error("Data formatting error")]
    BadFormat(#[from] SparseFormatError),
}

// Trait for updating P and A matrices from various data types
pub trait MatrixProblemDataUpdate<T: FloatT> {
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
    ) -> Result<(), SparseFormatError>;
}

// Trait for updating q and b vectors from various data types
pub trait VectorProblemDataUpdate<T: FloatT> {
    fn update_vector(&self, v: &mut [T], scale: &[T]) -> Result<(), SparseFormatError>;
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
    ) -> Result<(), DataUpdateError> {
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
    /// - an iterator zip(&index,&values), specifying a selective update of values.
    ///
    /// - an empty vector, in which case no action is taken.
    ///
    pub fn update_P<Data: MatrixProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_presolve_disabled()?;
        let d = &self.data.equilibration.d;
        data.update_matrix(&mut self.data.P, d, d)?;
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
    /// - an iterator zip(&index,&values), specifying a selective update of values.
    ///
    /// - an empty vector, in which case no action is taken.
    ///
    pub fn update_A<Data: MatrixProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_presolve_disabled()?;
        let d = &self.data.equilibration.d;
        let e = &self.data.equilibration.e;
        data.update_matrix(&mut self.data.A, e, d)?;
        // overwrite KKT data
        self.kktsystem.update_A(&self.data.A);
        Ok(())
    }

    /// Overwrites the `q` vector data in an existing solver object.  No action is taken if the input is empty.
    pub fn update_q<Data: VectorProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_presolve_disabled()?;
        let d = &self.data.equilibration.d;
        data.update_vector(&mut self.data.q, d)?;

        // flush unscaled norm. Will be recalculated during solve
        self.data.clear_normq();

        Ok(())
    }

    /// Overwrites the `b` vector data in an existing solver object.  No action is taken if the input is empty.
    pub fn update_b<Data: VectorProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_presolve_disabled()?;
        let e = &self.data.equilibration.e;
        data.update_vector(&mut self.data.b, e)?;

        // flush unscaled norm. Will be recalculated during solve
        self.data.clear_normb();

        Ok(())
    }

    fn check_presolve_disabled(&self) -> Result<(), DataUpdateError> {
        if self.settings.presolve_enable {
            Err(DataUpdateError::PresolveEnabled)
        } else {
            Ok(())
        }
    }
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

impl<T> MatrixProblemDataUpdate<T> for [T]
where
    T: FloatT,
{
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
    ) -> Result<(), SparseFormatError> {
        let data = self;
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

impl<T: FloatT> MatrixProblemDataUpdate<T> for Vec<T> {
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
    ) -> Result<(), SparseFormatError> {
        self.as_slice().update_matrix(M, lscale, rscale)
    }
}

impl<T: FloatT> MatrixProblemDataUpdate<T> for [T; 0] {
    fn update_matrix(
        &self,
        _M: &mut CscMatrix<T>,
        _lscale: &[T],
        _rscale: &[T],
    ) -> Result<(), SparseFormatError> {
        Ok(())
    }
}

// Can't write a single impl for [T], Vec<T> and [T;0] above because
// bounding by AsRef<[T]> is not specific enough to distinguish it from
// the zip iterator for partial updates implemented next.

impl<'a, T> MatrixProblemDataUpdate<T> for Zip<Iter<'a, usize>, Iter<'a, T>>
where
    T: FloatT,
{
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
    ) -> Result<(), SparseFormatError> {
        for (&idx, &value) in self.clone() {
            if idx >= M.nzval.len() {
                return Err(SparseFormatError::IncompatibleDimension);
            }
            let (row, col) = M.index_to_coord(idx);
            M.nzval[idx] = lscale[row] * rscale[col] * value;
        }
        Ok(())
    }
}

impl<T> VectorProblemDataUpdate<T> for [T]
where
    T: FloatT,
{
    fn update_vector(&self, v: &mut [T], scale: &[T]) -> Result<(), SparseFormatError> {
        let data = self;
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

impl<T: FloatT> VectorProblemDataUpdate<T> for Vec<T> {
    fn update_vector(&self, v: &mut [T], scale: &[T]) -> Result<(), SparseFormatError> {
        self.as_slice().update_vector(v, scale)
    }
}

impl<T: FloatT> VectorProblemDataUpdate<T> for [T; 0] {
    fn update_vector(&self, _v: &mut [T], _scale: &[T]) -> Result<(), SparseFormatError> {
        Ok(())
    }
}

// Can't write a single impl for [T], Vec<T> and [T;0] above because
// bounding by AsRef<[T]> is not specific enough to distinguish it from
// the zip iterator for partial updates implemented next.

impl<'a, T> VectorProblemDataUpdate<T> for Zip<Iter<'a, usize>, Iter<'a, T>>
where
    T: FloatT,
{
    fn update_vector(&self, v: &mut [T], scale: &[T]) -> Result<(), SparseFormatError> {
        for (&idx, &value) in self.clone() {
            if idx >= v.len() {
                return Err(SparseFormatError::IncompatibleDimension);
            }
            v[idx] = value * scale[idx];
        }
        Ok(())
    }
}
