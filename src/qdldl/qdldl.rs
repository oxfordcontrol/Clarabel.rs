#![allow(non_snake_case)]
use crate::algebra::*;
use core::cmp::{max, min};
use derive_builder::Builder;
use std::iter::zip;
use thiserror::Error;

/// Error codes returnable from [`QDLDLFactorisation`](QDLDLFactorisation) factor operations
#[derive(Error, Debug)]
pub enum QDLDLError {
    #[error("Matrix dimension fields are incompatible")]
    /// Matrix dimension fields are incompatible
    IncompatibleDimension,
    #[error("Matrix has a zero column")]
    /// Matrix has a zero column
    EmptyColumn,
    #[error("Matrix is not upper triangular")]
    /// Matrix is not upper triangular
    NotUpperTriangular,
    /// Matrix factorization produced a zero pivot
    #[error("Matrix factorization produced a zero pivot")]
    ZeroPivot,
    #[error("Invalid permutation vector")]
    /// Invalid permutation vector supplied
    InvalidPermutation,
}

#[derive(Builder, Debug, Clone)]
#[allow(missing_docs)]
/// Required settings for [`QDLDLFactorisation`](QDLDLFactorisation)
pub struct QDLDLSettings<T: FloatT> {
    /// "dense scale" parameter for AMD ordering
    #[builder(default = "1.0")]
    pub amd_dense_scale: f64,

    /// optional user-supplied custom permutation vector for the matrix
    #[builder(default = "None", setter(strip_option))]
    pub perm: Option<Vec<usize>>,

    /// Logical factorisation only, no numerical factorisation
    #[builder(default = "false")]
    pub logical: bool,

    /// optional user-supplied signs of the diagonal elements of D in LDL^T
    #[builder(default = "None", setter(strip_option))]
    pub Dsigns: Option<Vec<i8>>,

    /// Enable regularization during factorisation
    #[builder(default = "true")]
    pub regularize_enable: bool,

    /// Regularization epsilon parameter
    #[builder(default = "(1e-12).as_T()")]
    pub regularize_eps: T,

    /// Regularization delta parameter
    #[builder(default = "(1e-7).as_T()")]
    pub regularize_delta: T,
}

impl<T> Default for QDLDLSettings<T>
where
    T: FloatT,
{
    fn default() -> QDLDLSettings<T> {
        QDLDLSettingsBuilder::<T>::default().build().unwrap()
    }
}

/// Performs $LDL^T$ factorization of a symmetric quasidefinite matrix
#[derive(Debug)]
pub struct QDLDLFactorisation<T = f64> {
    /// permutation vector
    pub perm: Vec<usize>,
    /// inverse permutation
    #[allow(dead_code)] //Unused because we call ipermute in solve instead.  Keep anyway.
    iperm: Vec<usize>,
    /// lower triangular factor L in LDL^T
    pub L: CscMatrix<T>,
    /// vector of diagonal elements of D in LDL^T
    pub D: Vec<T>,
    /// vector of reciprocal diagonal elements of D in LDL^T
    pub Dinv: Vec<T>,
    /// internal workspace data
    workspace: QDLDLWorkspace<T>,
    /// true if factorisation is symbolic only
    is_symbolic: bool,
}

impl<T> QDLDLFactorisation<T>
where
    T: FloatT,
{
    /// create a new LDL^T factorisation
    pub fn new(
        Ain: &CscMatrix<T>,
        opts: Option<QDLDLSettings<T>>,
    ) -> Result<QDLDLFactorisation<T>, QDLDLError> {
        //sanity check on structure
        check_structure(Ain)?;
        _qdldl_new(Ain, opts)
    }

    /// returns the number of positive eigenvalues
    pub fn positive_inertia(&self) -> usize {
        self.workspace.positive_inertia
    }
    /// returns the number of regularisation shifts
    /// that were applied during factorisation
    pub fn regularize_count(&self) -> usize {
        self.workspace.regularize_count
    }

    /// Solves Ax = b using LDL factors for A.
    /// Solves in place (x replaces b)
    pub fn solve(&mut self, b: &mut [T]) {
        // bomb if logical factorisation only
        assert!(!self.is_symbolic);

        // bomb if b is the wrong size
        assert_eq!(b.len(), self.D.len());

        // permute b
        let tmp = &mut self.workspace.fwork;
        permute(tmp, b, &self.perm);

        //solve in place with tmp as permuted RHS
        _solve(
            &self.L.colptr,
            &self.L.rowval,
            &self.L.nzval,
            &self.Dinv,
            tmp,
        );

        // inverse permutation to put unpermuted soln in b
        ipermute(b, tmp, &self.perm);
    }

    /// Update a subset of the values of the matrix to be (re)factored.  See [`refactor`](crate::qdldl::QDLDLFactorisation::refactor)
    ///
    pub fn update_values(&mut self, indices: &[usize], values: &[T]) {
        let nzval = &mut self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt; //mapping from input matrix entries to triuA

        for (i, &idx) in indices.iter().enumerate() {
            nzval[AtoPAPt[idx]] = values[i];
        }
    }

    /// Update a subset of the values of the matrix to be (re)factored.  See [`refactor`](crate::qdldl::QDLDLFactorisation::refactor)
    ///
    pub fn scale_values(&mut self, indices: &[usize], scale: T) {
        let nzval = &mut self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt; //mapping from input matrix entries to triuA

        for &idx in indices.iter() {
            nzval[AtoPAPt[idx]] *= scale;
        }
    }

    /// Shifts a subset of the values of the matrix to be (re)factored.   The
    /// values are offset by `offset`, with `signs` a vector of +/- 1 values
    /// indicating the direction of shifts. See [`refactor`](crate::qdldl::QDLDLFactorisation::refactor)
    ///
    pub fn offset_values(&mut self, indices: &[usize], offset: T, signs: &[i8]) {
        assert_eq!(indices.len(), signs.len());

        let nzval = &mut self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt; //mapping from input matrix entries to triuA

        for (&idx, &sign) in zip(indices, signs) {
            match sign.signum() {
                1 => {
                    nzval[AtoPAPt[idx]] += offset;
                }
                -1 => {
                    nzval[AtoPAPt[idx]] -= offset;
                }
                _ => {}
            }
        }
    }

    /// Refactor a matrix after its data has been modified.   See [`update_values`](crate::qdldl::QDLDLFactorisation::update_values),
    /// [`scale_values`](crate::qdldl::QDLDLFactorisation::scale_values) and [`offset_values`](crate::qdldl::QDLDLFactorisation::offset_values)
    ///
    pub fn refactor(&mut self) -> Result<(), QDLDLError> {
        // It never makes sense to call refactor for a logical
        // factorization since it will always be the same.  Calling
        // this function implies that we want a numerical factorization
        self.is_symbolic = false;

        // The sparsity pattern of L was fixed by the factorization in
        // `new` (numeric or logical), so refactorizations can replay a
        // precomputed update schedule instead of re-deriving the pattern.
        // Built lazily here so that single-factorization use pays nothing.
        if matches!(self.workspace.schedule, ScheduleState::NotBuilt) {
            let ws = &self.workspace;
            self.workspace.schedule = match _build_schedule(
                &ws.triuA.colptr,
                &ws.triuA.rowval,
                &ws.etree,
                &self.L.colptr,
                &self.L.rowval,
            ) {
                Some(s) => ScheduleState::Ready(s),
                None => ScheduleState::Unavailable,
            };
        }

        if !matches!(self.workspace.schedule, ScheduleState::Ready(_)) {
            return _factor(
                &mut self.L,
                &mut self.D,
                &mut self.Dinv,
                &mut self.workspace,
                false,
            );
        }

        // destructure so the borrows of the schedule and the other
        // workspace fields are disjoint
        let QDLDLWorkspace {
            schedule,
            triuA,
            fwork,
            Dsigns,
            regularize_enable,
            regularize_eps,
            regularize_delta,
            regularize_count,
            positive_inertia,
            ..
        } = &mut self.workspace;
        let ScheduleState::Ready(sched) = schedule else {
            unreachable!()
        };

        *positive_inertia = _refactor_scheduled(
            sched,
            &triuA.colptr,
            &triuA.rowval,
            &triuA.nzval,
            &self.L.colptr,
            &self.L.rowval,
            &mut self.L.nzval,
            &mut self.D,
            &mut self.Dinv,
            fwork,
            Dsigns,
            *regularize_enable,
            *regularize_eps,
            *regularize_delta,
            regularize_count,
        )?;
        Ok(())
    }

    // test support: confirms refactorization used the scheduled replay path
    #[cfg(test)]
    pub(crate) fn refactor_schedule_is_ready(&self) -> bool {
        matches!(self.workspace.schedule, ScheduleState::Ready(_))
    }

    /// Returns the number of nonzeros in A for A = LDL^T
    pub fn nnzA(&self) -> usize {
        self.workspace.triuA.nnz()
    }

    /// Returns the number of nonzeros in L for A = LDL^T
    pub fn nnzL(&self) -> usize {
        self.L.nnz()
    }
}

fn check_structure<T: FloatT>(A: &CscMatrix<T>) -> Result<(), QDLDLError> {
    if !A.is_square() {
        return Err(QDLDLError::IncompatibleDimension);
    }

    if !A.is_triu() {
        return Err(QDLDLError::NotUpperTriangular);
    }

    //Error if A doesn't have at least one entry in every column
    if !A.colptr.windows(2).all(|c| c[0] < c[1]) {
        return Err(QDLDLError::EmptyColumn);
    }

    Ok(())
}

fn _qdldl_new<T: FloatT>(
    Ain: &CscMatrix<T>,
    opts: Option<QDLDLSettings<T>>,
) -> Result<QDLDLFactorisation<T>, QDLDLError> {
    let n = Ain.nrows();

    //get default values if no options passed at all
    let opts = opts.unwrap_or_default();

    //Use AMD ordering if a user-provided ordering
    //is not supplied.   For no ordering at all, the
    //user would need to pass (0..n).collect() explicitly
    let (perm, iperm);
    if let Some(_perm) = opts.perm {
        iperm = _invperm(&_perm)?;
        perm = _perm;
    } else {
        (perm, iperm, _) = get_amd_ordering(Ain, opts.amd_dense_scale);
    }

    //permute to (another) upper triangular matrix and store the
    //index mapping the input's entries to the permutation's entries
    let (A, AtoPAPt) = permute_symmetric(Ain, &iperm);

    // handle the (possibly permuted) vector of
    // diagonal D signs if one was specified.  Otherwise
    // otherwise all signs are positive
    let mut Dsigns = vec![1_i8; n];
    if let Some(ds) = opts.Dsigns {
        Dsigns = vec![1_i8; n];
        permute(&mut Dsigns, &ds, &perm);
    }

    // allocate workspace
    let mut workspace = QDLDLWorkspace::<T>::new(
        A,
        AtoPAPt,
        Dsigns,
        opts.regularize_enable,
        opts.regularize_eps,
        opts.regularize_delta,
    )?;

    //total nonzeros in factorization
    let sumLnz = workspace.Lnz.iter().sum();

    // allocate space for the L matrix row indices and data
    let mut L = CscMatrix::spalloc((n, n), sumLnz);

    // allocate for D and D inverse in LDL^T
    let mut D = vec![T::zero(); n];
    let mut Dinv = vec![T::zero(); n];

    // factor the matrix into A = LDL^T
    _factor(&mut L, &mut D, &mut Dinv, &mut workspace, opts.logical)?;

    Ok(QDLDLFactorisation {
        perm,
        iperm,
        L,
        D,
        Dinv,
        workspace,
        is_symbolic: opts.logical,
    })
}

#[derive(Debug)]
struct QDLDLWorkspace<T> {
    // internal workspace data
    etree: Vec<usize>,
    Lnz: Vec<usize>,
    iwork: Vec<usize>,
    bwork: Vec<bool>,
    fwork: Vec<T>,

    // number of positive values in D
    positive_inertia: usize,

    // The upper triangular matrix factorisation target
    // This is the post ordering PAPt of the original data
    triuA: CscMatrix<T>,

    // mapping from entries in the triu form
    // of the original input to the post ordering
    // triu form used for the factorization
    // this can be used when modifying entries
    // of the data matrix for refactoring
    AtoPAPt: Vec<usize>,

    //regularization signs and parameters
    Dsigns: Vec<i8>,
    regularize_enable: bool,
    regularize_eps: T,
    regularize_delta: T,

    // number of regularized entries in D
    regularize_count: usize,

    // precomputed control flow for numeric refactorization,
    // built lazily on the first refactor
    schedule: ScheduleState,
}

impl<T> QDLDLWorkspace<T>
where
    T: FloatT,
{
    pub fn new(
        triuA: CscMatrix<T>,
        AtoPAPt: Vec<usize>,
        Dsigns: Vec<i8>,
        regularize_enable: bool,
        regularize_eps: T,
        regularize_delta: T,
    ) -> Result<Self, QDLDLError> {
        let mut etree = vec![0; triuA.ncols()];
        let mut Lnz = vec![0; triuA.ncols()]; //nonzeros in each L column
        let mut iwork = vec![0; triuA.ncols() * 3];
        let bwork = vec![false; triuA.ncols()];
        let fwork = vec![T::zero(); triuA.ncols()];

        // compute elimination tree using QDLDL converted code
        _etree(
            triuA.nrows(),
            &triuA.colptr,
            &triuA.rowval,
            &mut iwork,
            &mut Lnz,
            &mut etree,
        )?;

        // positive inertia count.
        let positive_inertia = 0;

        // number of regularized entries in D. None to start
        let regularize_count = 0;

        Ok(Self {
            etree,
            Lnz,
            iwork,
            bwork,
            fwork,
            positive_inertia,
            triuA,
            AtoPAPt,
            Dsigns,
            regularize_enable,
            regularize_eps,
            regularize_delta,
            regularize_count,
            schedule: ScheduleState::NotBuilt,
        })
    }
}

// -------------------------------------
// Precomputed refactorization schedule
// -------------------------------------
//
// After the first factorization the sparsity pattern of L (colptr/rowval)
// is fixed; refactorizations change only numeric values.  `_factor_inner`
// nevertheless re-derives, for every column k, the list of prior columns
// that update it — an elimination-tree walk with marker arrays, a work
// buffer and a reversal step.  On matrices whose factors have little
// fill-in (typical for IPM KKT systems), that control-flow overhead
// dominates the actual arithmetic of a refactorization.
//
// `FactorSchedule` records the discovered control flow once, flat: for each
// column k in order, the sequence of (cidx, pos) update steps exactly as
// `_factor_inner` executes them, where cidx is the prior column supplying
// the update and pos is the position in L.nzval that receives L[k,cidx].
// The update range of a step is Lp[cidx]..pos: at the moment step (k,cidx)
// runs, column cidx holds exactly its entries with row index < k, because
// entries are appended to each column in increasing row order.
// `_refactor_scheduled` then replays the identical floating-point
// operations in the identical order — giving bit-identical L, D and
// regularization decisions — without re-deriving any of the pattern.
// This is the standard symbolic/numeric phase separation of sparse direct
// solvers (T. Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006,
// ch. 4); QDLDL's original single-phase design favours simplicity for
// one-shot factorizations, but Clarabel refactors the same pattern once
// per interior-point iteration.
//
// Indices are stored as u32 to halve the memory traffic of the replay.
// Patterns too large for that (nnz(L) or n >= 2^32) fall back to the
// original path via `ScheduleState::Unavailable`.

#[derive(Debug)]
struct FactorSchedule {
    // start of column k's steps in cidx/pos; length n+1
    colptr: Vec<u32>,
    // source column of each update step
    cidx: Vec<u32>,
    // position in L.nzval written by each update step
    pos: Vec<u32>,
}

#[derive(Debug)]
enum ScheduleState {
    NotBuilt,
    Unavailable,
    Ready(FactorSchedule),
}

// Records the update schedule by replaying the pattern-discovery phase of
// `_factor_inner` (which see), with the numeric work stripped out.  The two
// functions must stay in lockstep; as a defence, every recorded write
// position is verified against the already-computed pattern of L
// (Li[pos] == k), and the total step count against nnz(L).  Any mismatch
// returns None and refactorization falls back to `_factor_inner`.
fn _build_schedule(
    Ap: &[usize],
    Ai: &[usize],
    etree: &[usize],
    Lp: &[usize],
    Li: &[usize],
) -> Option<FactorSchedule> {
    let n = Lp.len() - 1;
    let nnzL = Lp[n];
    if n >= u32::MAX as usize || nnzL >= u32::MAX as usize {
        return None;
    }

    let mut colptr = Vec::with_capacity(n + 1);
    let mut cidx = Vec::with_capacity(nnzL);
    let mut pos = Vec::with_capacity(nnzL);

    let mut y_markers = vec![QDLDL_UNUSED; n];
    let mut y_idx = vec![0usize; n];
    let mut elim_buffer = vec![0usize; n];
    let mut next_colspace: Vec<usize> = Lp[0..n].to_vec();

    colptr.push(0u32);
    if n > 0 {
        colptr.push(0u32); // column 0 has no update steps
    }

    for k in 1..n {
        // pattern-discovery phase, exactly as in _factor_inner
        let mut nnz_y = 0;

        for &bidx in &Ai[Ap[k]..Ap[k + 1]] {
            if bidx == k {
                continue;
            }

            if y_markers[bidx] == QDLDL_UNUSED {
                y_markers[bidx] = QDLDL_USED;
                elim_buffer[0] = bidx;
                let mut nnz_e = 1;

                let mut next_idx = etree[bidx];
                while next_idx != QDLDL_UNKNOWN && next_idx < k {
                    if y_markers[next_idx] == QDLDL_USED {
                        break;
                    }
                    y_markers[next_idx] = QDLDL_USED;
                    elim_buffer[nnz_e] = next_idx;
                    next_idx = etree[next_idx];
                    nnz_e += 1;
                }

                while nnz_e != 0 {
                    nnz_e -= 1;
                    y_idx[nnz_y] = elim_buffer[nnz_e];
                    nnz_y += 1;
                }
            }
        }

        // record the value-placement phase of _factor_inner
        for i in (0..nnz_y).rev() {
            let c = y_idx[i];
            let p = next_colspace[c];

            // verify against the pattern L already has
            if Li[p] != k {
                return None;
            }

            cidx.push(c as u32);
            pos.push(p as u32);
            next_colspace[c] += 1;
            y_markers[c] = QDLDL_UNUSED;
        }
        colptr.push(cidx.len() as u32);
    }

    if cidx.len() != nnzL {
        return None;
    }

    Some(FactorSchedule { colptr, cidx, pos })
}

// Numeric refactorization by schedule replay.  Performs the identical
// floating point operations, in the identical order, as
// `_factor_inner(..., logical_factor = false)` on the same pattern, so the
// results (L, D, Dinv, inertia and regularization counts) are bit-identical
// to that function's.  See `_build_schedule` for the schedule's invariants.
#[allow(clippy::too_many_arguments)]
fn _refactor_scheduled<T: FloatT>(
    sched: &FactorSchedule,
    Ap: &[usize],
    Ai: &[usize],
    Ax: &[T],
    Lp: &[usize],
    Li: &[usize],
    Lx: &mut [T],
    D: &mut [T],
    Dinv: &mut [T],
    y_vals: &mut [T],
    Dsigns: &[i8],
    regularize_enable: bool,
    regularize_eps: T,
    regularize_delta: T,
    regularize_count: &mut usize,
) -> Result<usize, QDLDLError> {
    *regularize_count = 0;
    let mut positiveValuesInD = 0;
    let n = Lp.len() - 1;

    y_vals.fill(T::zero());
    D.fill(T::zero());

    // First element of the diagonal D, as in _factor_inner
    D[0] = Ax[0];
    if regularize_enable {
        let sign = T::from_i8(Dsigns[0]).unwrap();
        if D[0] * sign < regularize_eps {
            D[0] = regularize_delta * sign;
            *regularize_count += 1;
        }
    }
    if D[0].is_zero() {
        return Err(QDLDLError::ZeroPivot);
    }
    if D[0] > T::zero() {
        positiveValuesInD += 1;
    }
    Dinv[0] = T::recip(D[0]);

    for k in 1..n {
        // scatter the kth column of A above the diagonal into the sparse
        // accumulator, and initialize D[k], exactly as in _factor_inner
        for i in Ap[k]..Ap[k + 1] {
            let bidx = Ai[i];
            if bidx == k {
                D[k] = Ax[i];
            } else {
                y_vals[bidx] = Ax[i];
            }
        }

        // replay the update steps for this column
        let (f, l) = (sched.colptr[k] as usize, sched.colptr[k + 1] as usize);
        for (&c, &p) in zip(&sched.cidx[f..l], &sched.pos[f..l]) {
            let (cidx, tmp_idx) = (c as usize, p as usize);
            let y_vals_cidx = y_vals[cidx];

            let (f, l) = (Lp[cidx], tmp_idx);
            unsafe {
                // Safety: Li entries index the matrix dimension, and the
                // schedule's positions were verified against Li at build
                // time; both bound y_vals/Lx as in _factor_inner.
                for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                    *(y_vals.get_unchecked_mut(Lij)) -= Lxj * y_vals_cidx;
                }

                let Lx_tmp_idx = y_vals_cidx * *Dinv.get_unchecked(cidx);
                *Lx.get_unchecked_mut(tmp_idx) = Lx_tmp_idx;
                *D.get_unchecked_mut(k) -= y_vals_cidx * Lx_tmp_idx;
            }

            y_vals[cidx] = T::zero();
        }

        // pivot regularization / rejection, as in _factor_inner
        if regularize_enable {
            let sign = T::from_i8(Dsigns[k]).unwrap();
            if D[k] * sign < regularize_eps {
                D[k] = regularize_delta * sign;
                *regularize_count += 1;
            }
        }
        if D[k].is_zero() {
            return Err(QDLDLError::ZeroPivot);
        }
        if D[k] > T::zero() {
            positiveValuesInD += 1;
        }
        Dinv[k] = T::recip(D[k]);
    }

    Ok(positiveValuesInD)
}

fn _factor<T: FloatT>(
    L: &mut CscMatrix<T>,
    D: &mut [T],
    Dinv: &mut [T],
    workspace: &mut QDLDLWorkspace<T>,
    logical: bool,
) -> Result<(), QDLDLError> {
    if logical {
        L.nzval.fill(T::one());
        D.fill(T::one());
        Dinv.fill(T::one());
    }

    // factor using QDLDL C style converted code
    let A = &workspace.triuA;

    let pos_d_count = _factor_inner(
        A.n,
        &A.colptr,
        &A.rowval,
        &A.nzval,
        &mut L.colptr,
        &mut L.rowval,
        &mut L.nzval,
        D,
        Dinv,
        &workspace.Lnz,
        &workspace.etree,
        &mut workspace.bwork,
        &mut workspace.iwork,
        &mut workspace.fwork,
        logical,
        &workspace.Dsigns,
        workspace.regularize_enable,
        workspace.regularize_eps,
        workspace.regularize_delta,
        &mut workspace.regularize_count,
    )?;

    workspace.positive_inertia = pos_d_count;

    Ok(())
}

const QDLDL_UNKNOWN: usize = usize::MAX;
const QDLDL_USED: bool = true;
const QDLDL_UNUSED: bool = false;

// Compute the elimination tree for a quasidefinite matrix
// in compressed sparse column form.

fn _etree(
    n: usize,
    Ap: &[usize],
    Ai: &[usize],
    work: &mut [usize],
    Lnz: &mut [usize],
    etree: &mut [usize],
) -> Result<usize, QDLDLError> {
    // zero out Lnz and work.  Set all etree values to unknown
    work.fill(0);
    Lnz.fill(0);
    etree.fill(QDLDL_UNKNOWN);

    // compute the elimination tree
    for j in 0..n {
        work[j] = j;
        for istart in Ai.iter().take(Ap[j + 1]).skip(Ap[j]) {
            let mut i = *istart;

            while work[i] != j {
                if etree[i] == QDLDL_UNKNOWN {
                    etree[i] = j;
                }
                Lnz[i] += 1; // nonzeros in this column
                work[i] = j;
                i = etree[i];
            }
        }
    }

    Ok(0)
}

//allow too_many_arguments since this follows the implementation
//of the C version of QDLDL.
#[allow(clippy::too_many_arguments)]
fn _factor_inner<T: FloatT>(
    n: usize,
    Ap: &[usize],
    Ai: &[usize],
    Ax: &[T],
    Lp: &mut [usize],
    Li: &mut [usize],
    Lx: &mut [T],
    D: &mut [T],
    Dinv: &mut [T],
    Lnz: &[usize],
    etree: &[usize],
    bwork: &mut [bool],
    iwork: &mut [usize],
    fwork: &mut [T],
    logical_factor: bool,
    Dsigns: &[i8],
    regularize_enable: bool,
    regularize_eps: T,
    regularize_delta: T,
    regularize_count: &mut usize,
) -> Result<usize, QDLDLError> {
    *regularize_count = 0;
    let mut positiveValuesInD = 0;

    // partition working memory into pieces
    let y_markers = bwork;
    let (y_idx, iwork) = iwork.split_at_mut(n);
    let (elim_buffer, next_colspace) = iwork.split_at_mut(n);
    let y_vals = fwork;

    //set Lp to cumsum(Lnz), starting from zero
    Lp[0] = 0;
    let mut acc = 0;
    for (Lp, Lnz) in zip(&mut Lp[1..], Lnz) {
        acc += Lnz;
        *Lp = acc;
    }

    //  set all y_idx to be 'unused' initially
    // in each column of L, the next available space
    // to start is just the first space in the column
    y_markers.fill(QDLDL_UNUSED);
    y_vals.fill(T::zero());
    D.fill(T::zero());
    next_colspace.copy_from_slice(&Lp[0..Lp.len() - 1]);

    if !logical_factor {
        // First element of the diagonal D.
        D[0] = Ax[0];
        if regularize_enable {
            let sign = T::from_i8(Dsigns[0]).unwrap();
            if D[0] * sign < regularize_eps {
                D[0] = regularize_delta * sign;
                *regularize_count += 1;
            }
        }

        if D[0].is_zero() {
            return Err(QDLDLError::ZeroPivot);
        }
        if D[0] > T::zero() {
            positiveValuesInD += 1;
        }
        Dinv[0] = T::recip(D[0]);
    }

    // Start from second row (k=1) here. The upper LH corner is trivially 0
    // in L b/c we are only computing the subdiagonal elements
    for k in 1..n {
        // NB : For each k, we compute a solution to
        // y = L(0:(k-1),0:k-1))\b, where b is the kth
        // column of A that sits above the diagonal.
        // The solution y is then the kth row of L,
        // with an implied '1' at the diagonal entry.

        // number of nonzeros in this row of L
        let mut nnz_y = 0; // number of elements in this row

        // This loop determines where nonzeros
        // will go in the kth row of L, but doesn't
        // compute the actual values

        for i in Ap[k]..Ap[k + 1] {
            let bidx = Ai[i]; //we are working on this element of b

            // Initialize D[k] as the element of this column
            // corresponding to the diagonal place.  Don't use
            // this element as part of the elimination step
            // that computes the k^th row of L
            if bidx == k {
                D[k] = Ax[i];
                continue;
            }

            y_vals[bidx] = Ax[i]; // initialise y(bidx) = b(bidx)

            // use the forward elimination tree to figure
            // out which elements must be eliminated after
            // this element of b
            let next_idx = bidx;

            if y_markers[next_idx] == QDLDL_UNUSED {
                //this y term not already visited

                y_markers[next_idx] = QDLDL_USED; //I touched this one
                elim_buffer[0] = next_idx; // It goes at the start of the current list
                let mut nnz_e = 1; //length of unvisited elimination path from here

                let mut next_idx = etree[bidx];

                while next_idx != QDLDL_UNKNOWN && next_idx < k {
                    if y_markers[next_idx] == QDLDL_USED {
                        break;
                    }

                    y_markers[next_idx] = QDLDL_USED; // I touched this one
                    elim_buffer[nnz_e] = next_idx; // It goes in the current list
                    next_idx = etree[next_idx]; // one step further along tree
                    nnz_e += 1; // the list is one longer than before
                }

                // now put the buffered elimination list into
                // my current ordering in reverse order
                while nnz_e != 0 {
                    nnz_e -= 1;
                    y_idx[nnz_y] = elim_buffer[nnz_e];
                    nnz_y += 1;
                }
            }
        }

        // This for loop places nonzeros values in the k^th row
        for i in (0..nnz_y).rev() {
            // which column are we working on?
            let cidx = y_idx[i];

            // loop along the elements in this
            // column of L and subtract to solve to y
            let tmp_idx = next_colspace[cidx];

            // don't compute Lx for logical factorisation
            // this logic is not implemented in the C version
            if !logical_factor {
                let y_vals_cidx = y_vals[cidx];

                let (f, l) = (Lp[cidx], tmp_idx);
                unsafe {
                    //Safety : Here the Lij index comes from the rowval
                    //field of the sparse L factor matrix, and should
                    //always be bounded by the matrix dimension.
                    for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                        *(y_vals.get_unchecked_mut(Lij)) -= Lxj * y_vals_cidx;
                    }

                    // Now I have the cidx^th element of y = L\b.
                    // so compute the corresponding element of
                    // this row of L and put it into the right place
                    let Lx_tmp_idx = y_vals_cidx * *Dinv.get_unchecked(cidx);
                    *Lx.get_unchecked_mut(tmp_idx) = Lx_tmp_idx;
                    *D.get_unchecked_mut(k) -= y_vals_cidx * Lx_tmp_idx;
                }
            }

            // record which row it went into
            Li[tmp_idx] = k;
            next_colspace[cidx] += 1;

            // reset the y_vals and indices back to zero and QDLDL_UNUSED
            // once I'm done with them
            y_vals[cidx] = T::zero();
            y_markers[cidx] = QDLDL_UNUSED;
        }

        if !logical_factor {
            // apply dynamic regularization
            if regularize_enable {
                let sign = T::from_i8(Dsigns[k]).unwrap();
                if D[k] * sign < regularize_eps {
                    D[k] = regularize_delta * sign;
                    *regularize_count += 1;
                }
            }

            // Maintain a count of the positive entries
            // in D.  If we hit a zero, we can't factor
            // this matrix, so abort
            if D[k].is_zero() {
                return Err(QDLDLError::ZeroPivot);
            }
            if D[k] > T::zero() {
                positiveValuesInD += 1;
            }

            // compute the inverse of the diagonal
            Dinv[k] = T::recip(D[k]);
        }
    } //end for k

    Ok(positiveValuesInD)
}

// Solves (L+I)x = b, with x replacing b (with standard bounds checks)
fn _lsolve_safe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    for i in 0..x.len() {
        let xi = x[i];
        let (f, l) = (Lp[i], Lp[i + 1]);
        let Lx = &Lx[f..l];
        let Li = &Li[f..l];
        for (&Lij, &Lxj) in zip(Li, Lx) {
            x[Lij] -= Lxj * xi;
        }
    }
}

// Solves (L+I)'x = b, with x replacing b (with standard bounds checks)
fn _ltsolve_safe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    for i in (0..x.len()).rev() {
        let mut s = T::zero();
        let (f, l) = (Lp[i], Lp[i + 1]);
        let Lx = &Lx[f..l];
        let Li = &Li[f..l];
        for (&Lij, &Lxj) in zip(Li, Lx) {
            s += Lxj * x[Lij];
        }
        x[i] -= s;
    }
}

// -------------------------------------
// Versions of L\x and Lᵀ \ x that use unchecked indexing.
//
// Safety : The values in colptr array Lp at the time this
// function is reached should be bounded by the sizes of the
// arrays in Lx and Li.  The length of x should be compatible
// with the row index entries in Li
// -------------------------------------

// Solves (L+I)x = b, with x replacing b.  Unchecked version
fn _lsolve_unsafe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    unsafe {
        for i in 0..x.len() {
            let xi = *x.get_unchecked(i);
            let f = *Lp.get_unchecked(i);
            let l = *Lp.get_unchecked(i + 1);
            for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                *(x.get_unchecked_mut(Lij)) -= Lxj * xi;
            }
        }
    }
}

// Solves (L+I)'x = b, with x replacing b.  Unchecked version.
fn _ltsolve_unsafe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], x: &mut [T]) {
    unsafe {
        for i in (0..x.len()).rev() {
            let mut s = T::zero();
            let f = *Lp.get_unchecked(i);
            let l = *Lp.get_unchecked(i + 1);
            for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                s += Lxj * (*x.get_unchecked(Lij));
            }
            *x.get_unchecked_mut(i) -= s;
        }
    }
}

// Solves D(L+I)'x = b, with x replacing b.  Unchecked version.
fn _dltsolve_unsafe<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], Dinv: &[T], x: &mut [T]) {
    unsafe {
        for i in (0..x.len()).rev() {
            let mut s = T::zero();
            let f = *Lp.get_unchecked(i);
            let l = *Lp.get_unchecked(i + 1);
            for (&Lxj, &Lij) in zip(&Lx[f..l], &Li[f..l]) {
                s += Lxj * (*x.get_unchecked(Lij));
            }

            let xi = x.get_unchecked_mut(i);
            *xi *= *Dinv.get_unchecked(i);
            *xi -= s;
        }
    }
}

// Solves Ax = b where A has given LDL factors, with x replacing b
fn _solve<T: FloatT>(Lp: &[usize], Li: &[usize], Lx: &[T], Dinv: &[T], b: &mut [T]) {
    // We call the `unsafe`d version of the forward and backward substitution
    // functions here, since the matrix data should be well posed and x of
    // compatible dimensions.   For super safety or debugging purposes, there
    // are also `safe` versions implemented above.
    _lsolve_unsafe(Lp, Li, Lx, b);

    // combined D and L^T solve in one pass
    _dltsolve_unsafe(Lp, Li, Lx, Dinv, b);

    // in separate passes for reference
    //zip(b.iter_mut(), Dinv).for_each(|(b, d)| *b *= *d);
    //_ltsolve_unsafe(Lp, Li, Lx, b);
}

// Construct an inverse permutation from a permutation
fn _invperm(p: &[usize]) -> Result<Vec<usize>, QDLDLError> {
    let mut b = vec![0; p.len()];

    for (i, j) in p.iter().enumerate() {
        if *j < p.len() && b[*j] == 0 {
            b[*j] = i;
        } else {
            return Err(QDLDLError::InvalidPermutation);
        }
    }
    Ok(b)
}

// permutation and inverse permutation
// functions that require no allocation
// p must be a valid permutation vector
// in both cases for safety

pub(crate) fn permute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    debug_assert!(p.is_empty() || *p.iter().max().unwrap() < x.len());
    unsafe {
        zip(p, x).for_each(|(p, x)| *x = *b.get_unchecked(*p));
    }
}

pub(crate) fn ipermute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    debug_assert!(p.is_empty() || *p.iter().max().unwrap() < x.len());
    unsafe {
        zip(p, b).for_each(|(p, b)| *x.get_unchecked_mut(*p) = *b);
    }
}

// Given a sparse symmetric matrix `A` (with only upper triangular entries), return
// permuted sparse symmetric matrix `P` (also only upper triangular) given the
// inverse permutation vector `iperm`."
pub(crate) fn permute_symmetric<T: FloatT>(
    A: &CscMatrix<T>,
    iperm: &[usize],
) -> (CscMatrix<T>, Vec<usize>) {
    // perform a number of argument checks
    let (_m, n) = A.size();
    let mut P = CscMatrix::<T>::spalloc((n, n), A.nnz());

    // we will record a mapping of entries from A to PAPt
    let mut AtoPAPt = vec![0; A.nnz()];

    _permute_symmetric_inner(
        A,
        &mut AtoPAPt,
        iperm,
        &mut P.rowval,
        &mut P.colptr,
        &mut P.nzval,
    );
    (P, AtoPAPt)
}

// the main function without extra argument checks
// following the book: Timothy Davis - Direct Methods for Sparse Linear Systems

fn _permute_symmetric_inner<T: FloatT>(
    A: &CscMatrix<T>,
    AtoPAPt: &mut [usize],
    iperm: &[usize],
    Pr: &mut [usize],
    Pc: &mut [usize],
    Pv: &mut [T],
) {
    // 1. count number of entries that each column of P will have
    let n = A.nrows();
    let mut num_entries = vec![0; n];
    let Ar = &A.rowval;
    let Ac = &A.colptr;
    let Av = &A.nzval;

    // count the number of upper-triangle entries in columns of P,
    // keeping in mind the row permutation
    for colA in 0..n {
        let colP = iperm[colA];
        // loop over entries of A in column A...
        for rowA in Ar.iter().take(Ac[colA + 1]).skip(Ac[colA]) {
            let rowP = iperm[*rowA];
            // ...and check if entry is upper triangular
            if *rowA <= colA {
                // determine to which column the entry belongs after permutation
                let col_idx = max(rowP, colP);
                num_entries[col_idx] += 1;
            }
        }
    }

    // 2. calculate permuted Pc = P.colptr from number of entries
    // Pc is one longer than num_entries here.
    Pc[0] = 0;
    let mut acc = 0;
    for (Pckp1, ne) in zip(&mut Pc[1..], &num_entries) {
        *Pckp1 = acc + ne;
        acc = *Pckp1;
    }
    // reuse this memory to keep track of free entries in rowval
    num_entries.copy_from_slice(&Pc[0..n]);

    // use alias
    let mut row_starts = num_entries;

    // 3. permute the row entries and position of corresponding nzval
    for colA in 0..n {
        let colP = iperm[colA];
        // loop over rows of A and determine where each row entry of A should be stored
        for rowA_idx in Ac[colA]..Ac[colA + 1] {
            let rowA = Ar[rowA_idx];
            // check if upper triangular
            if rowA <= colA {
                let rowP = iperm[rowA];
                // determine column to store the entry
                let col_idx = max(colP, rowP);

                // find next free location in rowval (this results in unordered columns in the rowval)
                let rowP_idx = row_starts[col_idx];

                // store rowval and nzval
                Pr[rowP_idx] = min(colP, rowP);
                Pv[rowP_idx] = Av[rowA_idx];

                //record this into the mapping vector
                AtoPAPt[rowA_idx] = rowP_idx;

                // increment next free location
                row_starts[col_idx] += 1;
            }
        }
    }
}

pub(crate) fn get_amd_ordering<T: FloatT>(
    A: &CscMatrix<T>,
    amd_dense_scale: f64,
) -> (Vec<usize>, Vec<usize>, amd::Info) {
    // PJG: For interested readers - setting amd_dense_scale to 1.5 seems to work better
    // for KKT systems in QP problems, but this ad hoc method can surely be improved

    // computes a permutation for A using AMD default parameters
    let mut control = amd::Control::default();
    control.dense *= amd_dense_scale; //increase the default value
    let (perm, iperm, info) = amd::order(A.nrows(), &A.colptr, &A.rowval, &control).unwrap();
    (perm, iperm, info)
}

//configure tests of internals
#[path = "test.rs"]
#[cfg(test)]
mod test;
