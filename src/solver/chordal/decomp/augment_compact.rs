#![allow(non_snake_case)]
#![allow(clippy::too_many_arguments)]

// -----------------------------------------
// Functions related to clique tree based transformation (compact decomposition)
// see: Kim: Exploiting sparsity in linear and nonlinear matrix inequalities via
// positive semidefinite matrix completion (2011), p.53
// -----------------------------------------

use crate::solver::chordal::ConeMapEntry;
use crate::solver::chordal::SparsityPattern;
use crate::solver::core::cones::*;
use crate::{
    algebra::*,
    solver::{
        chordal::{ChordalInfo, SuperNodeTree, VertexSet},
        SupportedConeT::*,
    },
};
use std::cmp::{max, min};
use std::ops::Range;
use std::{collections::HashMap, iter::zip};

type BlockOverlapTriplet = (usize, usize, bool);

impl<T> ChordalInfo<T>
where
    T: FloatT,
{
    #[allow(clippy::type_complexity)]
    pub(crate) fn decomp_augment_compact(
        &mut self,
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
    ) -> (
        CscMatrix<T>,
        Vec<T>,
        CscMatrix<T>,
        Vec<T>,
        Vec<SupportedConeT<T>>,
    ) {
        let (A_new, b_new, cones_new) = self.find_compact_A_b_and_cones(A, b);

        // how many variables did we add?
        let nadd = A_new.n - A.n;

        let P_new = CscMatrix::blockdiag(&[P, &CscMatrix::zeros((nadd, nadd))]).unwrap();

        let mut q_new = vec![T::zero(); q.len() + nadd];
        q_new[0..q.len()].copy_from(q);

        (P_new, q_new, A_new, b_new, cones_new)
    }

    fn find_compact_A_b_and_cones(
        &mut self,
        A: &CscMatrix<T>,
        b: &[T],
    ) -> (CscMatrix<T>, Vec<T>, Vec<SupportedConeT<T>>) {
        // the cones that we used to form the decomposition
        let cones = &self.init_cones;

        // determine number of final augmented matrix and number of overlapping entries
        let (Aa_m, Aa_n, n_overlaps) = self.find_A_dimension(A);

        // allocate sparse components for the augmented A
        let Aa_nnz = A.nnz() + 2 * n_overlaps;
        let mut Aa_I = vec![usize::MAX; Aa_nnz]; //usize::MAX forces a panic if not overwritten
        let mut Aa_J = extra_columns(Aa_nnz, A.nnz(), A.n);
        let mut Aa_V = alternating_sequence::<T>(Aa_nnz, A.nnz());
        findnz(&mut Aa_J, &mut Aa_V, A);

        // allocate sparse components for the augmented b
        let bs = SparseVector::new(b);
        let mut ba_I = vec![usize::MAX; bs.nzval.len()]; //usize::MAX forces a panic if not overwritten

        // preallocate the decomposed cones and the mapping
        // from decomposed cones back to the originals
        let n_decomposed = self.final_cone_count();
        let mut cones_new = Vec::with_capacity(n_decomposed);
        let mut cone_maps = Vec::with_capacity(n_decomposed);

        // an enumerate-like mutable iterator for the patterns.  We will expand cones
        // assuming that they are non-decomposed until we reach an index that agrees
        // the internally stored orig_index of the next pattern.

        let mut patterns_iter = self.spatterns.iter().peekable();
        let mut patterns_count = 0..self.spatterns.len();
        let row_ranges = cones.rng_cones_iter();

        let mut row_ptr = 0; // index to start of next cone in A_I
        let mut overlap_ptr = A.nnz(); // index to next row for +1, -1 overlap entries

        for (coneidx, (cone, row_range)) in zip(cones, row_ranges).enumerate() {
            if patterns_iter.len() != 0 && patterns_iter.peek().unwrap().orig_index == coneidx {
                assert!(matches!(cone, SupportedConeT::PSDTriangleConeT(_)));
                (row_ptr, overlap_ptr) = add_entries_with_sparsity_pattern(
                    &mut Aa_I,
                    &mut ba_I,
                    &mut cones_new,
                    &mut cone_maps,
                    A,
                    &bs,
                    row_range,
                    patterns_iter.next().unwrap(),
                    patterns_count.next().unwrap(),
                    row_ptr,
                    overlap_ptr,
                );
            } else {
                (row_ptr, overlap_ptr) = add_entries_with_cone(
                    &mut Aa_I,
                    &mut ba_I,
                    &mut cones_new,
                    &mut cone_maps,
                    A,
                    &bs,
                    row_range,
                    cone,
                    row_ptr,
                    overlap_ptr,
                );
            }
        }

        // save the cone_maps for use when reconstructing
        // solution to the original problem
        self.cone_maps = Some(cone_maps);

        let A_new = CscMatrix::new_from_triplets(Aa_m, Aa_n, Aa_I, Aa_J, Aa_V);
        let b_new = SparseVector {
            nzind: ba_I,
            nzval: bs.nzval,
            n: Aa_m,
        }
        .into();

        (A_new, b_new, cones_new)
    }

    // find the dimension of the `compact' form `A' matrix and its number of overlaps

    fn find_A_dimension(&self, A: &CscMatrix<T>) -> (usize, usize, usize) {
        let (dim, num_overlaps) = self.get_decomposed_dim_and_overlaps();

        let rows = dim;
        let cols = A.n + num_overlaps;

        (rows, cols, num_overlaps)
    }
}

// Handles all cones that are not decomposed by a sparsity pattern
fn add_entries_with_cone<T>(
    Aa_I: &mut [usize],
    ba_I: &mut [usize],
    cones_new: &mut Vec<SupportedConeT<T>>,
    cone_maps: &mut Vec<ConeMapEntry>,
    A: &CscMatrix<T>,
    b: &SparseVector<T>,
    row_range: Range<usize>,
    cone: &SupportedConeT<T>,
    row_ptr: usize,
    overlap_ptr: usize,
) -> (usize, usize)
where
    T: FloatT,
{
    let n = A.n;
    let offset = (row_ptr as isize) - (row_range.start as isize);

    // populate b
    let row_range_col = get_rows_vec(b, row_range.clone());
    if let Some(row_range_col) = row_range_col {
        for k in row_range_col {
            ba_I[k] = b.nzind[k].checked_add_signed(offset).unwrap();
        }
    }

    // populate A
    for col in 0..n {
        // indices that store the rows in column col in A
        let row_range_col = get_rows_mat(A, col, row_range.clone());
        if let Some(row_range_col) = row_range_col {
            for k in row_range_col {
                Aa_I[k] = A.rowval[k].checked_add_signed(offset).unwrap();
            }
        }
    }

    // here we make a copy of the cone
    cones_new.push(cone.clone());

    // since this cone is standalone and not decomposed, the index
    // of its origin cone must be either one more than the previous one,
    // or 0 if it's the first

    let orig_index = {
        if cone_maps.is_empty() {
            0
        } else {
            cone_maps.last().unwrap().orig_index + 1
        }
    };

    cone_maps.push(ConeMapEntry {
        orig_index,
        tree_and_clique: None,
    });

    (row_ptr + cone.nvars(), overlap_ptr)
}

// Handle decomposable cones with a SparsityPattern. The row vectors A_I and b_I
// have to be edited in such a way that entries of one clique appear contiguously.
fn add_entries_with_sparsity_pattern<T>(
    A_I: &mut [usize],
    b_I: &mut [usize],
    cones_new: &mut Vec<SupportedConeT<T>>,
    cone_maps: &mut Vec<ConeMapEntry>,
    A: &CscMatrix<T>,
    b: &SparseVector<T>,
    row_range: Range<usize>,
    spattern: &SparsityPattern,
    spattern_index: usize,
    row_ptr: usize,
    overlap_ptr: usize,
) -> (usize, usize)
where
    T: FloatT,
{
    let mut row_ptr = row_ptr;
    let mut overlap_ptr = overlap_ptr;
    let sntree = &spattern.sntree;
    let ordering = &spattern.ordering;

    let (_, n) = A.size();

    // determine the row ranges for each of the subblocks
    let clique_to_rows = clique_rows_map(row_ptr, sntree);

    // loop over cliques in descending topological order
    for i in (0..sntree.n_cliques).rev() {
        // get supernodes and separators and undo the reordering
        // NB: these are now Vector, not VertexSet
        let mut separator: Vec<usize> = sntree
            .get_separators(i)
            .iter()
            .map(|&v| spattern.ordering[v])
            .collect();
        let mut snode: Vec<usize> = sntree
            .get_snode(i)
            .iter()
            .map(|&v| spattern.ordering[v])
            .collect();
        separator.sort();
        snode.sort();

        // compute sorted block indices (i, j, flag) for this clique with an
        // information flag whether an entry (i, j) is an overlap
        let block_indices = get_block_indices(&snode, &separator, ordering.len());

        // If we encounter an overlap with a parent clique we have to be able to find the
        // location of the overlapping entry. Therefore load and reorder the parent clique
        let parent_rows;
        let mut parent_clique;
        if i == (sntree.n_cliques - 1) {
            parent_rows = 0..0;
            parent_clique = vec![];
        } else {
            let parent_index = sntree.get_clique_parent(i);
            parent_rows = clique_to_rows.get(&parent_index).unwrap().clone();
            parent_clique = get_clique_by_index(sntree, parent_index)
                .iter()
                .map(|&v| spattern.ordering[v])
                .collect();
            parent_clique.sort();
        }

        // Loop over all the columns and shift the rows in A_I and b_I according to the clique structure
        // Here we just convert to empty ranges rather than trying to carry Option<Range>> all the way
        // down the call stack.   Done for consistency with COSMO

        for col in 0..n {
            let row_range_col = get_rows_mat(A, col, row_range.clone()).unwrap_or(0..0);

            let row_range_b = {
                if col == 0 {
                    get_rows_vec(b, row_range.clone()).unwrap_or(0..0)
                } else {
                    0..0
                }
            };

            overlap_ptr = add_clique_entries(
                A_I,
                b_I,
                &A.rowval,
                &b.nzind,
                &block_indices,
                &parent_clique,
                parent_rows.clone(),
                col,
                row_ptr,
                overlap_ptr,
                row_range.clone(),
                row_range_col.clone(),
                row_range_b.clone(),
            );
        }

        // create new PSD cones for the subblocks, and tag them
        // with their tree and clique number

        let cone_dim = sntree.get_nblk(i);
        cones_new.push(PSDTriangleConeT(cone_dim));
        cone_maps.push(ConeMapEntry {
            orig_index: spattern.orig_index,
            tree_and_clique: Some((spattern_index, i)),
        });
        row_ptr += triangular_number(cone_dim);
    }

    (row_ptr, overlap_ptr)
}

// Loop over all entries (i, j) in the clique and either set the correct row in `A_I` and `b_I`
// if (i, j) is not an overlap,or add an overlap column with (-1 and +1) in the correct positions.

fn add_clique_entries(
    A_I: &mut [usize],
    b_I: &mut [usize],
    A_rowval: &[usize],
    b_nzind: &[usize],
    block_indices: &[BlockOverlapTriplet],
    parent_clique: &[usize],
    parent_rows: Range<usize>,
    col: usize,
    row_ptr: usize,
    overlap_ptr: usize,
    row_range: Range<usize>,
    row_range_col: Range<usize>,
    row_range_b: Range<usize>,
) -> usize {
    let mut overlap_ptr = overlap_ptr;

    for (counter, &block_idx) in block_indices.iter().enumerate() {
        let new_row_val = row_ptr + counter;
        let (i, j, is_overlap) = block_idx;

        // a block index that corresponds to an overlap
        if is_overlap {
            if col == 0 {
                // this creates the +1 entry
                A_I[overlap_ptr] = new_row_val;
                // this creates the -1 entry
                A_I[overlap_ptr + 1] =
                    parent_rows.start + parent_block_indices(parent_clique, i, j);
                overlap_ptr += 2;
            }
        } else {
            let k = coord_to_upper_triangular_index((i, j));
            modify_clique_rows(
                A_I,
                k,
                A_rowval,
                new_row_val,
                row_range.clone(),
                row_range_col.clone(),
            );
            if col == 0 {
                modify_clique_rows(
                    b_I,
                    k,
                    b_nzind,
                    new_row_val,
                    row_range.clone(),
                    row_range_b.clone(),
                );
            }
        }
    }
    overlap_ptr
}

// Given the nominal entry position `k = linearindex(i, j)` find and modify with `new_row_val`
// the actual location of that entry in the global row vector `rowval`.

fn modify_clique_rows(
    v: &mut [usize],
    k: usize,
    rowval: &[usize],
    new_row_val: usize,
    row_range: Range<usize>,
    row_range_col: Range<usize>,
) {
    let row_0 = get_row_index(k, rowval, row_range, row_range_col);

    // row_0 = None happens when (i, j) references an edge that was added by merging cliques,
    // and we can disregard it.
    if let Some(row_0) = row_0 {
        v[row_0] = new_row_val;
    }
}

// Given the svec index `k` and an offset `row_range_col.start`, return the location of the
// (i, j)th entry in the row vector `rowval`.
// Julia returns 0 on failure here b/c it is 1-indexed.   Here we return Option<usize> instead
// to allow for zero indexing while keeping usize types.

fn get_row_index(
    k: usize,
    rowval: &[usize],
    row_range: Range<usize>,
    row_range_col: Range<usize>,
) -> Option<usize> {
    if row_range_col.clone().eq(0..0) {
        return None;
    }

    let k_shift = row_range.start + k;

    // determine upper set boundary of where the row could be
    let u = min(row_range_col.end, row_range_col.start + k_shift + 1);

    // find index of first entry >= k, starting in the interval [l, u]
    // if no, entry is >= k, r should be > u

    let l = row_range_col.start;
    let r = rowval[l..u].partition_point(|&y| y < k_shift) + l;

    // if no r s.t. rowval[r] = k_shift was found, that means that the
    // (i, j)th entry represents an edded edge (zero) from clique merging
    if r >= u || rowval[r] != k_shift {
        None
    } else {
        Some(r)
    }
}

// Find the index of k=svec(i, j) in the parent clique `par_clique`.#

fn parent_block_indices(parent_clique: &[usize], i: usize, j: usize) -> usize {
    let ir = parent_clique.partition_point(|&x| x < i); //first index >= i
    let jr = parent_clique.partition_point(|&x| x < j); //first index >= j
    coord_to_upper_triangular_index((ir, jr))
}

// Given a cliques supernodes and separators, compute all the indices (i, j) of the corresponding matrix block
// in the format (i, j, flag), where flag is equal to false if entry (i, j) corresponds to an overlap of the
// clique and true otherwise.

//`nv` is the number of vertices in the graph that we are trying to decompose.

fn get_block_indices(snode: &[usize], separator: &[usize], nv: usize) -> Vec<BlockOverlapTriplet> {
    let N = separator.len() + snode.len();

    let mut block_indices = Vec::<BlockOverlapTriplet>::with_capacity(triangular_number(N));

    for &j in separator.iter() {
        for &i in separator.iter() {
            if i <= j {
                block_indices.push((i, j, true));
            }
        }
    }

    for &j in snode.iter() {
        for &i in snode.iter() {
            if i <= j {
                block_indices.push((i, j, false));
            }
        }
    }

    for &i in snode {
        for &j in separator {
            block_indices.push((min(i, j), max(i, j), false));
        }
    }

    block_indices.sort_by_cached_key(|x| x.1 * nv + x.0);

    block_indices
}

// Return the row ranges of each clique after the decomposition, shifted by `row_start`.

fn clique_rows_map(row_start: usize, sntree: &SuperNodeTree) -> HashMap<usize, Range<usize>> {
    let n_cliques = sntree.n_cliques;
    let mut row_start = row_start;

    let mut out = HashMap::with_capacity(n_cliques);

    for i in (0..n_cliques).rev() {
        let num_rows = triangular_number(sntree.get_nblk(i));
        let rows = row_start..(row_start + num_rows);
        let indx = sntree.snode_post[i];
        out.insert(indx, rows);
        row_start += num_rows;
    }

    out
}

fn get_rows_subset(rows: &[usize], row_range: Range<usize>) -> Option<Range<usize>> {
    if rows.is_empty() || row_range.is_empty() {
        return None;
    }

    if *rows.last().unwrap() < row_range.start {
        return None;
    }

    if *rows.first().unwrap() >= row_range.end {
        return None;
    }

    let s = rows.partition_point(|&i| i < row_range.start);
    let e = rows.partition_point(|&i| i < row_range.end);

    Some(s..e)
}

fn get_rows_vec<T>(b: &SparseVector<T>, row_range: Range<usize>) -> Option<Range<usize>>
where
    T: FloatT,
{
    get_rows_subset(&b.nzind, row_range)
}

fn get_rows_mat<T>(A: &CscMatrix<T>, col: usize, row_range: Range<usize>) -> Option<Range<usize>>
where
    T: FloatT,
{
    let colrange = A.colptr[col]..(A.colptr[col + 1]);
    let rows = &A.rowval[colrange.clone()];
    let se = get_rows_subset(rows, row_range);

    se.map(|se| (colrange.start + se.start)..(colrange.start + se.end))
}

// Returns the appropriate amount of memory for `A.nzval`, including, starting from `n_start`,
// the (+1 -1) entries for the overlaps.

fn alternating_sequence<T>(total_length: usize, n_start: usize) -> Vec<T>
where
    T: FloatT,
{
    let mut v = vec![T::one(); total_length];
    for i in ((n_start + 1)..v.len()).step_by(2) {
        v[i] = -T::one();
    }
    v
}

// Returns the appropriate amount of memory for the columns of the augmented problem matrix `A`,
// including, starting from `n_start`, the columns for the (+1 -1) entries for the overlaps.

fn extra_columns(total_length: usize, n_start: usize, start_val: usize) -> Vec<usize> {
    let mut v = vec![0; total_length];
    let mut start_val = start_val;
    for i in (n_start..(v.len() - 1)).step_by(2) {
        v[i] = start_val;
        v[i + 1] = start_val;
        start_val += 1;
    }
    v
}

// Given sparse matrix components, write the columns and non-zero values into the first `numnz` entries of `J` and `V`.
fn findnz<T>(J: &mut [usize], V: &mut [T], S: &CscMatrix<T>)
where
    T: FloatT,
{
    let mut count = 0;
    for col in 0..S.n {
        for k in S.colptr[col]..S.colptr[col + 1] {
            J[count] = col;
            V[count] = S.nzval[k];
            count += 1
        }
    }
}

// Intentionally defined here separately from the other SuperNodeTree functions.
// This returns a clique directly from index i, rather than accessing the
// snode and separators via the postordering.   Only used in the compact
// problem decomposition functions

fn get_clique_by_index(sntree: &SuperNodeTree, i: usize) -> VertexSet {
    let mut out = VertexSet::new();
    out.extend(&sntree.snode[i]);
    out.extend(&sntree.separators[i]);
    out
}

#[test]
fn test_alternating_sequence() {
    let Annz = 2;
    let n_overlaps = 2;
    let Aa_nnz = Annz + 2 * n_overlaps;
    let Aa_V = alternating_sequence::<f64>(Aa_nnz, Annz);

    assert_eq!(Aa_V, vec![1., 1., 1., -1., 1., -1.]);
}
