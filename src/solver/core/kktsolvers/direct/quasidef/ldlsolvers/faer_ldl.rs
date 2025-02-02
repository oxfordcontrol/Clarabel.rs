#![allow(non_snake_case)]

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
// use faer::{
//     dyn_stack::{GlobalPodBuffer, PodStack, StackReq},
//     linalg::cholesky::ldlt::compute::LdltRegularization,
//     sparse::{
//         linalg::{amd::Control, cholesky::*, SupernodalThreshold},
//         SparseColMatRef, SymbolicSparseColMatRef,
//     },
//     Conj, Parallelism, Side,
// };
use faer::{
    linalg::cholesky::ldlt::factor::{LdltParams, LdltRegularization},
    sparse::{
        linalg::{
            amd::Control,
            cholesky::{
                factorize_symbolic_cholesky, CholeskySymbolicParams, LdltRef, SymbolicCholesky,
                SymmetricOrdering,
            },
            SupernodalThreshold,
        },
        SparseColMatRef, SymbolicSparseColMatRef,
    },
    Conj, MatMut, Par, Side, Spec,
};

use crate::algebra::*;
use crate::solver::core::kktsolvers::direct::DirectLDLSolver;
use crate::solver::core::CoreSettings;
use std::iter::zip;

#[derive(Debug)]

struct FaerLDLRegularizerParams<T: FloatT> {
    regularize: bool,
    eps: T,
    delta: T,
}

impl<'a, T> FaerLDLRegularizerParams<T>
where
    T: FloatT,
{
    fn to_faer(&self, Dsigns: &'a [i8]) -> LdltRegularization<'a, T> {
        let option = if self.regularize { Some(Dsigns) } else { None };
        LdltRegularization {
            dynamic_regularization_signs: option,
            dynamic_regularization_delta: self.delta,
            dynamic_regularization_epsilon: self.eps,
        }
    }
}

pub struct FaerDirectLDLSolver<T: FloatT> {
    // we will pre-permute the KKT matrix and factor it inside faer with
    // an identity permutation.   This means we will also need to store
    // the mapping from the original ordering to the permuted ordering
    // for both the KKT matrix and the Dsigns, and will need to permute
    // the rhs and solution vectors accordingly
    perm: Vec<usize>,
    iperm: Vec<usize>,

    // permuted KKT matrix
    perm_kkt: CscMatrix<T>,
    perm_map: Vec<usize>,
    perm_dsigns: Vec<i8>,

    // space for permuted LHS/RHS when solving
    bperm: Vec<T>,

    // regularizer parameters captured from settings
    regularizer_params: FaerLDLRegularizerParams<T>,
    ldlt_params: Spec<LdltParams, T>,

    // symbolic + numeric cholesky data
    symbolic_cholesky: SymbolicCholesky<usize>,
    ld_vals: Vec<T>,

    // workspace for faer factor/solve calls
    work: MemBuffer,

    parallelism: Par,
}

impl<T> FaerDirectLDLSolver<T>
where
    T: FloatT,
{
    pub fn nthreads_from_settings(setting: usize) -> usize {
        faer::utils::thread::parallelism_degree(faer::Par::rayon(setting))
    }

    pub fn new(KKT: &CscMatrix<T>, Dsigns: &[i8], settings: &CoreSettings<T>) -> Self {
        assert!(KKT.is_square(), "KKT matrix is not square");

        // -----------------------------

        // Par::rayon(0) here is equivalent to rayon::current_num_threads()
        let parallelism = {
            match settings.max_threads {
                0 => Par::rayon(0),
                1 => Par::Seq,
                _ => Par::rayon(settings.max_threads as usize),
            }
        };

        // manually compute an AMD ordering for the KKT matrix
        // and permute it to match the ordering used in QDLDL
        let amd_dense_scale = 1.5; // magic number from QDLDL
        let (perm, iperm) = crate::qdldl::get_amd_ordering(KKT, amd_dense_scale);

        let (mut perm_kkt, mut perm_map) = crate::qdldl::permute_symmetric(KKT, &iperm);

        //Permute the Dsigns to match the ordering to be used internally
        let mut perm_dsigns = vec![1_i8; Dsigns.len()];
        permute(&mut perm_dsigns, Dsigns, &perm);

        // the amd_params will not be used by faer even though
        // we put them into CholeskySymbolicParams, provided
        // that my ordering is SymmetricOrdering::Identity

        let amd_params = Control {
            ..Default::default()
        };

        let supernodal_flop_ratio_threshold = SupernodalThreshold::AUTO;
        let cholesky_params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold,
            amd_params,
            ..Default::default()
        };
        // -----------------------------

        sort_csc_columns_with_map(&mut perm_kkt, &mut perm_map);

        let symbKKT = SymbolicSparseColMatRef::new_checked(
            perm_kkt.n,
            perm_kkt.n,
            &perm_kkt.colptr,
            None,
            &perm_kkt.rowval,
        );

        let symbolic_cholesky = factorize_symbolic_cholesky(
            symbKKT,
            Side::Upper,
            SymmetricOrdering::Identity,
            cholesky_params,
        )
        .unwrap();

        let ld_vals = vec![T::zero(); symbolic_cholesky.len_val()];

        let regularizer_params = FaerLDLRegularizerParams {
            regularize: settings.dynamic_regularization_enable,
            delta: settings.dynamic_regularization_delta,
            eps: settings.dynamic_regularization_eps,
        };

        let ldlt_params: Spec<LdltParams, T> = Default::default();

        // Required workspace for faer factor and solve
        let req_factor =
            symbolic_cholesky.factorize_numeric_ldlt_scratch::<T>(parallelism, ldlt_params);
        let req_solve = symbolic_cholesky.solve_in_place_scratch::<T>(1, parallelism); // 1 is the number of RHS
        let req = StackReq::any_of(&[req_factor, req_solve]);
        let work = MemBuffer::new(req);

        let bperm = vec![T::zero(); perm_kkt.n];

        Self {
            perm,
            iperm,
            perm_kkt,
            perm_map,
            perm_dsigns,
            bperm,
            regularizer_params,
            ldlt_params,
            symbolic_cholesky,
            ld_vals,
            work,
            parallelism,
        }
    }
}

impl<T> DirectLDLSolver<T> for FaerDirectLDLSolver<T>
where
    T: FloatT,
{
    fn update_values(&mut self, index: &[usize], values: &[T]) {
        // PJG: this is replicating the update_values function in qdldl
        let nzval = &mut self.perm_kkt.nzval; // post perm internal data
        let AtoPAPt = &self.perm_map; //mapping from input matrix entries

        for (i, &idx) in index.iter().enumerate() {
            nzval[AtoPAPt[idx]] = values[i];
        }
    }

    fn scale_values(&mut self, index: &[usize], scale: T) {
        // PJG: this is replicating the scale_values function in qdldl
        let nzval = &mut self.perm_kkt.nzval; // post perm internal data
        let AtoPAPt = &self.perm_map; //mapping from input matrix entries

        for &idx in index.iter() {
            nzval[AtoPAPt[idx]] *= scale;
        }
    }

    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]) {
        // PJG: this is replicating the offset_values function in qdldl
        let nzval = &mut self.perm_kkt.nzval; // post perm internal data
        let AtoPAPt = &self.perm_map; //mapping from input matrix entries

        for (&idx, &sign) in zip(index, signs) {
            let sign: T = T::from_i8(sign).unwrap();
            nzval[AtoPAPt[idx]] += offset * sign;
        }
    }

    fn solve(&mut self, _kkt: &CscMatrix<T>, x: &mut [T], b: &[T]) {
        // NB: faer solves in place.  Permute b to match the ordering used internally
        permute(&mut self.bperm, b, &self.perm);

        let rhs = MatMut::from_column_major_slice_mut(&mut self.bperm[0..], b.len(), 1);
        let ldlt = LdltRef::new(&self.symbolic_cholesky, self.ld_vals.as_slice());

        ldlt.solve_in_place_with_conj(
            Conj::No,
            rhs,
            self.parallelism,
            MemStack::new(&mut self.work),
        );

        // bperm is now the solution, permute it back to the original ordering
        permute(x, &self.bperm, &self.iperm);
    }

    fn refactor(&mut self, _kkt: &CscMatrix<T>) -> bool {
        let symbKKT = SymbolicSparseColMatRef::new_checked(
            self.perm_kkt.n,
            self.perm_kkt.n,
            &self.perm_kkt.colptr,
            None,
            &self.perm_kkt.rowval,
        );

        let a: SparseColMatRef<usize, T> =
            SparseColMatRef::new(symbKKT, self.perm_kkt.nzval.as_slice());

        let regularizer = self.regularizer_params.to_faer(&self.perm_dsigns);

        self.symbolic_cholesky
            .factorize_numeric_ldlt(
                self.ld_vals.as_mut_slice(),
                a,
                Side::Upper,
                regularizer,
                self.parallelism,
                MemStack::new(&mut self.work),
                self.ldlt_params,
            )
            .is_ok() // PJG: convert to bool for consistency with qdldl.   Should really return Result here and elsewhere
    }

    fn required_matrix_shape() -> MatrixTriangle {
        MatrixTriangle::Triu
    }
}

// ---------------------------------------------------------------------
// utility functions
// ---------------------------------------------------------------------

fn sort_csc_columns_with_map<T>(M: &mut CscMatrix<T>, map: &mut [usize])
where
    T: FloatT,
{
    // when we permute the KKT matrix into an upper triangular form, the
    // data within each column is no long ordered by increasing row number.

    // Fix that here, and update the map to reflect the new ordering

    // we need to sort on the inverted map, then invert back
    let mut imap = invperm(map);

    // it is not obvious to me how to sort all three of the vectors
    // M.nzvals, Mrowvals, and map together without allocating.
    // Allocate a permutation vector for the whole thing, get sort
    // indices for the rowvals columns, then permute everything
    let mut perm = (0..M.nnz()).collect::<Vec<_>>();

    for col in 0..M.n {
        let start = M.colptr[col];
        let end = M.colptr[col + 1];
        perm[start..end].sort_by_key(|&k| M.rowval[k]);
    }

    let mut tmpT = vec![T::zero(); M.nzval.len()];
    let mut tmpint = vec![0usize; M.nzval.len()];

    tmpT.copy_from_slice(&M.nzval);
    permute(&mut M.nzval, &tmpT, &perm);

    tmpint.copy_from_slice(&M.rowval);
    permute(&mut M.rowval, &tmpint, &perm);

    tmpint.copy_from_slice(&imap);
    permute(&mut imap, &tmpint, &perm);

    // map = invperm(imap), but in place
    for (i, j) in imap.iter().enumerate() {
        map[*j] = i;
    }

    M.check_format().unwrap();
}

// ---------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------

#[test]
fn test_faer_ldl() {
    let KKT = CscMatrix {
        m: 6,
        n: 6,
        colptr: vec![0, 1, 2, 4, 6, 8, 10],
        rowval: vec![0, 1, 0, 2, 1, 3, 0, 4, 1, 5],
        nzval: vec![1.0, 2.0, 1.0, -1.0, 1.0, -2.0, -1.0, -3.0, -1.0, -4.0],
    };

    let Dsigns = vec![1, 1, -1, -1, -1, -1];

    let mut solver = FaerDirectLDLSolver::new(&KKT, &Dsigns, &CoreSettings::default());

    let mut x = vec![0.0; 6];
    let b = vec![1.0, 2.0, 3.0, 4., 5., 6.];

    // check that the permuted values are in what should be natural order
    let nzval = &solver.perm_kkt.nzval; // post perm internal data
    let AtoPAPt = &solver.perm_map; //mapping from input matrix entries

    for i in 0..nzval.len() {
        assert_eq!(KKT.nzval[i], nzval[AtoPAPt[i]]);
    }

    solver.refactor(&KKT);
    solver.solve(&KKT, &mut x, &b);

    let xsol = vec![
        1.0,
        0.9090909090909091,
        -2.0,
        -1.5454545454545454,
        -2.0,
        -1.7272727272727275,
    ];
    assert!(x.norm_inf_diff(&xsol) < 1e-10);

    // assign a new entry at the end of the KKT matrix and resolve
    solver.update_values(&[9], &[-10.0]);

    solver.refactor(&KKT);
    solver.solve(&KKT, &mut x, &b);

    let xsol = vec![
        1.0,
        1.3076923076923077,
        -2.0,
        -1.346153846153846,
        -2.0,
        -0.7307692307692306,
    ];
    assert!(x.norm_inf_diff(&xsol) < 1e-10);

    // scale and update everything for codecov.
    solver.offset_values(&[1, 2], 3., &[1, -1]);
    solver.scale_values(&[1, 2], 2.);
}

#[test]
fn test_faer_qp() {
    use crate::solver::core::solver::IPSolver;
    use crate::solver::SolverStatus;

    let P = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![2.0],
    };
    let q = [1.0];
    let A = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![-1.0],
    };
    let b = [-2.0];
    let cones = vec![crate::solver::SupportedConeT::NonnegativeConeT(1)];

    let settings = crate::solver::DefaultSettingsBuilder::default()
        .direct_solve_method("faer".to_owned())
        .build()
        .unwrap();

    let mut solver = crate::solver::DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
}
