#![allow(non_snake_case)]

// -----------------------------------------
// Functions related to `traditional' decomposition
// -----------------------------------------

use crate::{
    algebra::*,
    solver::{
        chordal::{ChordalInfo, SparsityPattern},
        SupportedConeT::{self, *},
    },
};

impl<T> ChordalInfo<T>
where
    T: FloatT,
{
    #[allow(clippy::type_complexity)]
    pub(crate) fn decomp_augment_standard(
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
        // allocate H and new decomposed cones.  H will be stored
        // in chordal_info and is required for the reversal step

        let (H, cones_new) = self.find_standard_H_and_cones();

        let P_new = CscMatrix::blockdiag(&[P, &CscMatrix::<T>::zeros((H.n, H.n))]).unwrap();

        let mut q_new = vec![T::zero(); q.len() + H.n];
        q_new[0..q.len()].copy_from(q);

        let mut negI = CscMatrix::identity(H.ncols());
        negI.negate();
        let Z = CscMatrix::zeros((H.n, A.n));

        let A_new = CscMatrix::hvcat(&[&[A, &H], &[&Z, &negI]]).unwrap();

        let mut b_new = vec![T::zero(); b.len() + H.n];
        b_new[0..b.len()].copy_from(b);

        // save the H matrix for use when reconstructing
        // solution to the original problem
        self.H = Some(H);

        (P_new, q_new, A_new, b_new, cones_new)
    }

    // Find the transformation matrix `H` and its associated cones for the standard decomposition.

    fn find_standard_H_and_cones(&mut self) -> (CscMatrix<T>, Vec<SupportedConeT<T>>) {
        // the cones that we used to form the decomposition
        let cones = &self.init_cones;

        // preallocate H and new decomposed cones
        let lenH = self.find_H_col_dimension();
        let mut H_I = Vec::with_capacity(lenH);

        // ncones from decomposition, plus one for an additional equality constraint
        let mut cones_new = Vec::<SupportedConeT<T>>::with_capacity(self.final_cone_count() + 1);

        // +1 cone count above is for this equality constraint
        let (_, m) = self.init_dims;
        cones_new.push(ZeroConeT(m));

        // an iterator for the patterns.  We will expand cones assuming that
        // they are non-decomposed until we reach an index that agrees with
        // the internally stored coneidx of the next pattern.
        let mut patterns_iter = self.spatterns.iter().peekable();
        let mut row = 0;

        for (coneidx, cone) in cones.iter().enumerate() {
            if patterns_iter.len() != 0 && patterns_iter.peek().unwrap().orig_index == coneidx {
                assert!(matches!(cone, SupportedConeT::PSDTriangleConeT(_)));
                decompose_with_sparsity_pattern(
                    &mut H_I,
                    &mut cones_new,
                    patterns_iter.next().unwrap(),
                    row,
                );
            } else {
                decompose_with_cone(&mut H_I, &mut cones_new, cone, row);
            }

            row += cone.nvars();
        }

        let Hdims = (row, lenH); //one entry per column
        let H = CscMatrix::<T>::new_from_triplets(
            Hdims.0,
            Hdims.1,
            H_I,
            (0usize..lenH).collect(),
            vec![T::one(); lenH],
        );

        (H, cones_new)
    }

    fn find_H_col_dimension(&self) -> usize {
        let (cols, _) = self.get_decomposed_dim_and_overlaps();
        cols
    }
}

fn decompose_with_cone<T>(
    H_I: &mut Vec<usize>,
    cones_new: &mut Vec<SupportedConeT<T>>,
    cone: &SupportedConeT<T>,
    row: usize,
) where
    T: FloatT,
{
    for i in 0..cone.nvars() {
        H_I.push(row + i);
    }

    cones_new.push(cone.clone());
}

fn decompose_with_sparsity_pattern<T>(
    H_I: &mut Vec<usize>,
    cones_new: &mut Vec<SupportedConeT<T>>,
    spattern: &SparsityPattern,
    row: usize,
) where
    T: FloatT,
{
    let sntree = &spattern.sntree;

    for i in 0..sntree.n_cliques {
        let clique = sntree.get_clique(i);

        // the graph and tree algorithms determined the clique vertices of an
        // AMD-permuted matrix. Since the location of the data hasn't changed
        // in reality, we have to map the clique vertices back
        let mut c: Vec<usize> = clique.iter().map(|&v| spattern.ordering[v]).collect();
        c.sort();

        add_subblock_map(H_I, &c, row);

        // add a new cone for this subblock
        let cdim = sntree.get_nblk(i);
        cones_new.push(PSDTriangleConeT(cdim));
    }
}

fn add_subblock_map(H_I: &mut Vec<usize>, clique_vertices: &[usize], row_start: usize) {
    let v = clique_vertices;

    for j in 0..v.len() {
        for i in 0..=j {
            let row = coord_to_upper_triangular_index((v[i], v[j]));
            H_I.push(row_start + row);
        }
    }
}
