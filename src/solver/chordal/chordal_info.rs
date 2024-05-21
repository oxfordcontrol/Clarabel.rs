#![allow(non_snake_case)]

use std::iter::zip;

// PJG import here for ConeRanges is weird.   Why is it
// not coming with SupportedConeT?
use crate::{
    algebra::*,
    qdldl::*,
    solver::{
        core::cones::ConeRanges,
        CoreSettings,
        SupportedConeT::{self, *},
    },
};

use super::SparsityPattern;

// -------------------------------------
// Chordal Decomposition Information
// -------------------------------------
#[derive(Debug)]
pub(crate) struct ConeMapEntry {
    pub orig_index: usize,
    pub tree_and_clique: Option<(usize, usize)>,
}
#[derive(Debug)]
pub(crate) struct ChordalInfo<T> {
    // sketch of the original problem
    pub init_dims: (usize, usize), // (n,m) dimensions of the original problem
    pub init_cones: Vec<SupportedConeT<T>>, // original cones of the problem

    // decomposed problem data
    pub spatterns: Vec<SparsityPattern>, // sparsity patterns for decomposed cones

    // `H' matrix for the standard chordal problem transformation
    // remains as nothing if the `compact' transform is used
    pub H: Option<CscMatrix<T>>,

    // mapping from each generated cone to its original cone
    // index, plus its tree and clique information if it has
    // been generated as part of a chordal decomposition
    // remains as nothing if the `standard' transform is used
    pub cone_maps: Option<Vec<ConeMapEntry>>,
}

impl<T> ChordalInfo<T>
where
    T: FloatT,
{
    pub(crate) fn new(
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: &CoreSettings<T>,
    ) -> Self {
        // initial problem data
        let init_dims = (A.ncols(), A.nrows());

        let mut chordal_info = Self {
            init_dims,
            init_cones: vec![],
            spatterns: vec![],
            H: None,         // no H to start
            cone_maps: None, // no cone_maps to start
        };

        chordal_info.find_sparsity_patterns(
            A,
            b,
            cones,
            settings.chordal_decomposition_merge_method.as_str(),
        );

        // Only copy the generating cones if we have decomposition,
        // since otherwise this object is going to be dropped anyway
        if chordal_info.is_decomposed() {
            chordal_info.init_cones = cones.to_vec();
        }

        chordal_info
    }

    fn find_sparsity_patterns(
        &mut self,
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        merge_method: &str,
    ) {
        let rng_cones = cones.rng_cones_iter();

        // aggregate sparsity pattern across the rows of [A;b]
        let mut nz_mask = find_aggregate_sparsity_mask(A, b);

        // find the sparsity patterns of the PSD cones
        for (coneidx, (cone, rowrange)) in zip(cones, rng_cones).enumerate() {
            if let PSDTriangleConeT(dim) = cone {
                self.analyse_psdtriangle_sparsity_pattern(
                    &mut nz_mask[rowrange],
                    *dim,
                    coneidx,
                    merge_method,
                );
            }
        }
    }

    fn analyse_psdtriangle_sparsity_pattern(
        &mut self,
        nz_mask: &mut [bool],
        conedim: usize,
        coneidx: usize,
        merge_method: &str,
    ) {
        // Force the diagonal entries to be marked, otherwise
        // the symbolic LDL step will fail.
        for i in 0..conedim {
            nz_mask[triangular_index(i)] = true;
        }

        if nz_mask.iter().all(|x| *x) {
            return; //dense / decomposable
        }

        let (L, ordering) = find_graph(nz_mask);

        let spattern = SparsityPattern::new(L, ordering, coneidx, merge_method);

        if spattern.sntree.n_cliques == 1 {
            return; // not decomposed, or everything re-merged
        }

        self.spatterns.push(spattern);
    }

    // did any PSD cones get decomposed?
    pub(crate) fn is_decomposed(&self) -> bool {
        !self.spatterns.is_empty()
    }

    // total number of cones we started with
    pub(crate) fn init_cone_count(&self) -> usize {
        self.init_cones.len()
    }

    // total number of cones we started with
    pub(crate) fn init_psd_cone_count(&self) -> usize {
        self.init_cones
            .iter()
            .filter(|c| matches!(c, SupportedConeT::PSDTriangleConeT(_)))
            .count()
    }

    pub(crate) fn final_cone_count(&self) -> usize {
        self.init_cone_count() + self.final_psd_cones_added()
    }

    pub(crate) fn final_psd_cone_count(&self) -> usize {
        self.init_psd_cone_count() + self.final_psd_cones_added()
    }

    pub(crate) fn premerge_psd_cone_count(&self) -> usize {
        self.init_psd_cone_count() + self.premerge_psd_cones_added()
    }

    pub(crate) fn decomposable_cone_count(&self) -> usize {
        self.spatterns.len()
    }

    pub(crate) fn final_psd_cones_added(&self) -> usize {
        // sum the number of cliques in each spattern
        let ncliques = self
            .spatterns
            .iter()
            .fold(0, |acc, pattern| acc + pattern.sntree.n_cliques);
        let ndecomposable = self.decomposable_cone_count();

        // subtract npatterns to avoid double counting the
        // original decomposed cones
        ncliques - ndecomposable
    }

    pub(crate) fn premerge_psd_cones_added(&self) -> usize {
        // sum the number of cliques in each spattern
        let ncones = self
            .spatterns
            .iter()
            .fold(0, |acc, pattern| acc + pattern.sntree.snode.len());
        let ndecomposable = self.decomposable_cone_count();

        // subtract npatterns to avoid double counting the
        // original decomposed cones
        ncones - ndecomposable
    }

    /*
     */
    pub(crate) fn get_decomposed_dim_and_overlaps(&self) -> (usize, usize) {
        let cones = &self.init_cones;
        let mut sum_cols = 0;
        let mut sum_overlaps = 0;
        let mut patterns_iter = self.spatterns.iter().peekable();

        for (coneidx, cone) in cones.iter().enumerate() {
            let (cols, overlap) = {
                match patterns_iter.peek() {
                    Some(pattern) if pattern.orig_index == coneidx => patterns_iter
                        .next()
                        .unwrap()
                        .sntree
                        .get_decomposed_dim_and_overlaps(),
                    _ => (cone.nvars(), 0),
                }
            };
            sum_cols += cols;
            sum_overlaps += overlap;
        }

        (sum_cols, sum_overlaps)
    }
}

// -------------------------------------
// utility functions
// -------------------------------------

// returns true in every row in which [A;b] has a nonzero

fn find_aggregate_sparsity_mask<T: FloatT>(A: &CscMatrix<T>, b: &[T]) -> Vec<bool> {
    let mut active = vec![false; b.len()];

    for &r in A.rowval.iter() {
        active[r] = true;
    }

    for (i, &bi) in b.iter().enumerate() {
        if bi != T::zero() {
            active[i] = true;
        }
    }
    active
}

fn find_graph(nz_mask: &[bool]) -> (CscMatrix<f64>, Vec<usize>) {
    let nz = nz_mask.iter().filter(|&x| *x).count();
    let mut rows = Vec::with_capacity(nz);
    let mut cols = Vec::with_capacity(nz);

    // check final row/col to get matrix dimension
    let (m, n) = upper_triangular_index_to_coord(nz_mask.len() - 1);
    let (m, n) = (m + 1, n + 1);
    assert_eq!(m, n);

    for (linearidx, &isnonzero) in nz_mask.iter().enumerate() {
        if isnonzero {
            let (row, col) = upper_triangular_index_to_coord(linearidx);
            rows.push(row);
            cols.push(col);
        }
    }

    // QDLDL doesn't currently allow for logical-only decomposition
    // on a matrix of Bools, so pattern must be a Float64 matrix here
    let vals = vec![1f64; rows.len()];
    let pattern = CscMatrix::new_from_triplets(m, n, rows, cols, vals);

    let opts = QDLDLSettingsBuilder::default()
        .logical(true)
        .build()
        .unwrap();

    let factors = QDLDLFactorisation::<f64>::new(&pattern, Some(opts)).unwrap();

    let mut L = factors.L;
    let ordering = factors.perm;

    // this takes care of the case that QDLDL returns an unconnected adjacency matrix L
    connect_graph(&mut L);

    (L, ordering)
}

fn connect_graph<T: FloatT>(L: &mut CscMatrix<T>) {
    // unconnected blocks don't have any entries below the diagonal in their right-most columns
    let n = L.ncols();

    for j in 0..(n - 1) {
        let row_val = &L.rowval;
        let col_ptr = &L.colptr;

        let mut connected = false;
        for &row in &row_val[col_ptr[j]..col_ptr[j + 1]] {
            if row > j {
                connected = true;
                break;
            }
        }

        // this insertion can happen in a midrange column, as long as
        // that column is the last one for a given block
        if !connected {
            L.set_entry((j + 1, j), T::one());
        }
    }
}
