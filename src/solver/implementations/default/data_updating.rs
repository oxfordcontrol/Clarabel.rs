#![allow(non_snake_case)]
use crate::algebra::*;
use crate::solver::DefaultSolver;
use core::iter::{zip, Zip};
use core::slice::Iter;
use thiserror::Error;

/// Error type returned by user data update utilities, e.g. [`check_format`](crate::algebra::CscMatrix::check_format) utility.
#[derive(Error, Debug)]
pub enum DataUpdateError {
    #[error("Data updates are not allowed when presolver is active")]
    /// Data updates are not allowed when presolver is active
    PresolveIsActive,
    #[cfg(feature = "sdp")]
    #[error("Data updates are not allowed when chordal decomposition is active")]
    /// Data updates are not allowed when chordal decomposition is active
    ChordalDecompositionIsActive,
    #[error("Data formatting error")]
    /// Data formatting error.   See [`SparseFormatError`]
    BadFormat(#[from] SparseFormatError),
}

/// Trait for updating problem data matrices (`P` and `A`) from various data types
pub trait MatrixProblemDataUpdate<T: FloatT> {
    /// Update matrix entries using associated left/right conditioners and scaling terms
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError>;
}

/// Trait for updating problem data vectors (`q`` and `b`) from various data types
pub trait VectorProblemDataUpdate<T: FloatT> {
    /// Update vector entries using associated elementwise and overall scaling terms
    fn update_vector(
        &self,
        v: &mut [T],
        vscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError>;
}

impl<T> DefaultSolver<T>
where
    T: FloatT,
{
    // PJG: rustdoc fails to resolve links to `update_P`, `update_q`, `update_A`, `update_b` below

    /// Overwrites internal problem data structures in a solver object with new data, avoiding new memory allocations.   
    /// See `update_P`, `update_q`, `update_A`, `update_b` for allowable inputs.
    ///
    /// <div class="warning">
    ///
    /// data updating functions will return an error when either presolving or chordal
    /// decomposition have modfied the original problem structure.  In order to guarantee
    /// that data updates will be accepted regardless of the original problem data, set
    /// `presolve_enable = false` and `chordal_decomposition_enable = false` in the solver settings.
    /// See also `is_data_update_allowed()`.
    ///
    /// </div>
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
    /// - a nonempty `Vec`, in which case the nonzero values of the original `P` are overwritten, preserving the sparsity pattern, or
    ///
    /// - a `CscMatrix`, in which case the input must match the sparsity pattern of the upper triangular part of the original `P`.
    ///
    /// - an iterator `zip(&index,&values)`, specifying a selective update of values.
    ///
    /// - an empty vector, in which case no action is taken.
    ///
    pub fn update_P<Data: MatrixProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_data_update_allowed()?;
        let d = &self.data.equilibration.d;
        let c = self.data.equilibration.c;
        data.update_matrix(&mut self.data.P, d, d, Some(c))?;
        // overwrite KKT data
        self.kktsystem.update_P(&self.data.P);
        Ok(())
    }

    /// Overwrites the `A` matrix data in an existing solver object.   The input `A` can be    
    ///
    /// - a nonempty `Vec`, in which case the nonzero values of the original `A` are overwritten, preserving the sparsity pattern, or
    ///
    /// - a `CscMatrix`, in which case the input must match the sparsity pattern of the original `A`.  
    ///
    /// - an iterator `zip(&index,&values)`, specifying a selective update of values.
    ///
    /// - an empty vector, in which case no action is taken.
    ///
    pub fn update_A<Data: MatrixProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_data_update_allowed()?;
        let d = &self.data.equilibration.d;
        let e = &self.data.equilibration.e;
        data.update_matrix(&mut self.data.A, e, d, None)?;
        // overwrite KKT data
        self.kktsystem.update_A(&self.data.A);
        Ok(())
    }

    /// Overwrites the `q` vector data in an existing solver object.  No action is taken if the input is empty.
    pub fn update_q<Data: VectorProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_data_update_allowed()?;
        let d = &self.data.equilibration.d;
        let c = self.data.equilibration.c;
        data.update_vector(&mut self.data.q, d, Some(c))?;

        // flush unscaled norm. Will be recalculated during solve
        self.data.clear_normq();

        Ok(())
    }

    /// Overwrites the `b` vector data in an existing solver object.  No action is taken if the input is empty.
    pub fn update_b<Data: VectorProblemDataUpdate<T>>(
        &mut self,
        data: &Data,
    ) -> Result<(), DataUpdateError> {
        self.check_data_update_allowed()?;
        let e = &self.data.equilibration.e;
        data.update_vector(&mut self.data.b, e, None)?;

        // flush unscaled norm. Will be recalculated during solve
        self.data.clear_normb();

        Ok(())
    }

    fn check_data_update_allowed(&self) -> Result<(), DataUpdateError> {
        if self.data.is_presolved() {
            return Err(DataUpdateError::PresolveIsActive);
        }
        #[cfg(feature = "sdp")]
        if self.data.is_chordal_decomposed() {
            return Err(DataUpdateError::ChordalDecompositionIsActive);
        }
        Ok(())
    }

    /// Returns `true` if problem structure has been modified by
    /// presolving or chordal decomposition
    pub fn is_data_update_allowed(&self) -> bool {
        self.check_data_update_allowed().is_ok()
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
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        self.check_equal_sparsity(M)?;
        let v = &self.nzval;
        v.update_matrix(M, lscale, rscale, cscale)
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
        cscale: Option<T>,
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
        if let Some(c) = cscale {
            M.scale(c);
        }

        Ok(())
    }
}

impl<T: FloatT> MatrixProblemDataUpdate<T> for Vec<T> {
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        self.as_slice().update_matrix(M, lscale, rscale, cscale)
    }
}

impl<T: FloatT> MatrixProblemDataUpdate<T> for [T; 0] {
    fn update_matrix(
        &self,
        _M: &mut CscMatrix<T>,
        _lscale: &[T],
        _rscale: &[T],
        _cscale: Option<T>,
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
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        for (&idx, &value) in self.clone() {
            if idx >= M.nzval.len() {
                return Err(SparseFormatError::IncompatibleDimension);
            }
            let (row, col) = M.index_to_coord(idx);
            if let Some(c) = cscale {
                M.nzval[idx] = lscale[row] * rscale[col] * c * value;
            } else {
                M.nzval[idx] = lscale[row] * rscale[col] * value;
            }
        }
        Ok(())
    }
}

impl<T> MatrixProblemDataUpdate<T> for (Vec<usize>, Vec<T>)
where
    T: FloatT,
{
    fn update_matrix(
        &self,
        M: &mut CscMatrix<T>,
        lscale: &[T],
        rscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        let z = zip(self.0.iter(), self.1.iter());
        z.update_matrix(M, lscale, rscale, cscale)
    }
}

impl<T> VectorProblemDataUpdate<T> for [T]
where
    T: FloatT,
{
    fn update_vector(
        &self,
        v: &mut [T],
        vscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        let data = self;
        if data.is_empty() {
            return Ok(());
        }

        if data.len() != v.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        v.copy_from_slice(data);

        //reapply original equilibration
        v.hadamard(vscale);

        if let Some(c) = cscale {
            v.scale(c);
        }

        Ok(())
    }
}

impl<T: FloatT> VectorProblemDataUpdate<T> for Vec<T> {
    fn update_vector(
        &self,
        v: &mut [T],
        vscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        self.as_slice().update_vector(v, vscale, cscale)
    }
}

impl<T: FloatT> VectorProblemDataUpdate<T> for [T; 0] {
    fn update_vector(
        &self,
        _v: &mut [T],
        _vscale: &[T],
        _cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
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
    fn update_vector(
        &self,
        v: &mut [T],
        vscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        for (&idx, &value) in self.clone() {
            if idx >= v.len() {
                return Err(SparseFormatError::IncompatibleDimension);
            }
            if let Some(c) = cscale {
                v[idx] = value * vscale[idx] * c;
            } else {
                v[idx] = value * vscale[idx];
            }
        }
        Ok(())
    }
}

impl<T> VectorProblemDataUpdate<T> for (Vec<usize>, Vec<T>)
where
    T: FloatT,
{
    fn update_vector(
        &self,
        v: &mut [T],
        vscale: &[T],
        cscale: Option<T>,
    ) -> Result<(), SparseFormatError> {
        let z = zip(self.0.iter(), self.1.iter());
        z.update_vector(v, vscale, cscale)
    }
}
