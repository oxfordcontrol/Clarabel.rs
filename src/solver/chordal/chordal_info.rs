#![allow(non_snake_case)]

use std::iter::zip;

// PJG import here fpr ConeRanges is weird.   Why is it
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
pub(crate) struct ConeMapEntry {
    pub orig_index: usize,
    pub tree_and_clique: Option<(usize, usize)>,
}

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
            &A,
            &b,
            &cones,
            &settings.chordal_decomposition_merge_method.as_str(),
        );

        // Only copy the generating cones if we have decomposition,
        // since otherwise this object is going to be dropped anyway
        if chordal_info.is_decomposed() {
            chordal_info.init_cones = cones.to_vec();
        }

        return chordal_info;
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
        let mut nz_mask = find_aggregate_sparsity_mask(&A, &b);

        // find the sparsity patterns of the PSD cones
        for (coneidx, (cone, rowrange)) in zip(cones, rng_cones).enumerate() {
            match cone {
                PSDTriangleConeT(dim) => {
                    self.analyse_psdtriangle_sparsity_pattern(
                        &mut nz_mask[rowrange],
                        *dim,
                        coneidx,
                        merge_method,
                    );
                }
                _ => {} //skip
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

        let (L, ordering) = find_graph(&nz_mask);

        let spattern = SparsityPattern::new(L, ordering, coneidx, &merge_method);

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
    fn init_cone_count(&self) -> usize {
        self.init_cones.len()
    }

    // Determine the total number of sets `num_total` after decomposition
    // and the number of new psd cones `num_new_psd_cones`.
    pub(crate) fn post_cone_count(&self) -> usize {
        // sum the number of cliques in each spattern
        let npatterns = self.spatterns.len();
        let ncliques = self
            .spatterns
            .iter()
            .fold(0, |acc, pattern| acc + pattern.sntree.n_cliques);

        // subtract npatterns to avoid double counting the
        // original decomposed cones
        return self.init_cone_count() - npatterns + ncliques;
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

    return (L, ordering);
}

fn connect_graph<T: FloatT>(L: &mut CscMatrix<T>) {
    // unconnected blocks don't have any entries below the diagonal in their right-most columns
    let n = L.ncols();

    for j in 0..(n - 1) {
        let row_val = &L.rowval;
        let col_ptr = &L.colptr;

        let mut connected = false;
        for k in col_ptr[j]..col_ptr[j + 1] {
            if row_val[k] > j {
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
