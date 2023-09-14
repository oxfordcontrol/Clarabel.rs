#![allow(non_snake_case)]

use super::datamaps::*;
use crate::algebra::*;
use crate::solver::core::cones::CompositeCone;
use crate::solver::core::cones::*;
use num_traits::Zero;

pub(crate) fn allocate_kkt_Hsblocks<T, Z>(cones: &CompositeCone<T>) -> Vec<Z>
where
    T: FloatT,
    Z: Zero + Clone,
{
    let mut nnz = 0;
    if let Some(rng_last) = cones.rng_blocks.last() {
        nnz = rng_last.end;
    }
    vec![Z::zero(); nnz]
}

pub(crate) fn assemble_kkt_matrix<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    shape: MatrixTriangle,
) -> (CscMatrix<T>, LDLDataMap) {
    let mut map = LDLDataMap::new(P, A, cones);
    let (m, n) = A.size();
    let p = map.sparse_maps.pdim();

    // entries actually on the diagonal of P
    let nnz_diagP = P.count_diagonal_entries();

    // total entries in the Hs blocks
    let nnz_Hsblocks = map.Hsblocks.len();

    let nnzKKT = P.nnz() +      // Number of elements in P
    n -                         // Number of elements in diagonal top left block
    nnz_diagP +                 // remove double count on the diagonal if P has entries
    A.nnz() +                   // Number of nonzeros in A
    nnz_Hsblocks +              // Number of elements in diagonal below A'
    map.sparse_maps.nnz_vec() +  // Number of elements in sparse cone off diagonals
    p; //Number of elements in diagonal of sparse cones

    let mut K = CscMatrix::<T>::spalloc((m + n + p, m + n + p), nnzKKT);

    _kkt_assemble_colcounts(&mut K, P, A, cones, &map, shape);
    _kkt_assemble_fill(&mut K, P, A, cones, &mut map, shape);

    (K, map)
}
fn _kkt_assemble_colcounts<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    map: &LDLDataMap,
    shape: MatrixTriangle,
) {
    let (m, n) = A.size();

    // use K.p to hold nnz entries in each
    // column of the KKT matrix
    K.colptr.fill(0);

    match shape {
        MatrixTriangle::Triu => {
            K.colcount_block(P, 0, MatrixShape::N);
            K.colcount_missing_diag(P, 0);
            K.colcount_block(A, n, MatrixShape::T);
        }
        MatrixTriangle::Tril => {
            K.colcount_missing_diag(P, 0);
            K.colcount_block(P, 0, MatrixShape::T);
            K.colcount_block(A, 0, MatrixShape::N);
        }
    }

    // track the next sparse column to fill (assuming triu fill)
    let mut pcol = m + n; //next sparse column to fill
    let mut sparse_map_iter = map.sparse_maps.iter();

    for (i, cone) in cones.iter().enumerate() {
        let row = cones.rng_cones[i].start + n;

        // add the Hs blocks in the lower right
        let blockdim = cone.numel();
        if cone.Hs_is_diagonal() {
            K.colcount_diag(row, blockdim);
        } else {
            K.colcount_dense_triangle(row, blockdim, shape);
        }

        //add sparse expansions columns for sparse cones
        if cone.is_sparse_expandable() {
            let sc = cone.to_sparse_expansion().unwrap();
            let thismap = sparse_map_iter.next().unwrap();
            sc.csc_colcount_sparsecone(thismap, K, row, pcol, shape);
            pcol += thismap.pdim();
        }
    }
}

fn _kkt_assemble_fill<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    map: &mut LDLDataMap,
    shape: MatrixTriangle,
) {
    let (m, n) = A.size();

    // cumsum total entries to convert to K.p
    K.colcount_to_colptr();

    match shape {
        MatrixTriangle::Triu => {
            K.fill_block(P, &mut map.P, 0, 0, MatrixShape::N);
            K.fill_missing_diag(P, 0); // after adding P, since triu form
                                       // fill in value for A, top right (transposed/rowwise)
            K.fill_block(A, &mut map.A, 0, n, MatrixShape::T);
        }
        MatrixTriangle::Tril => {
            K.fill_missing_diag(P, 0); // before adding P, since tril form
            K.fill_block(P, &mut map.P, 0, 0, MatrixShape::T);
            // fill in value for A, bottom left (not transposed)
            K.fill_block(A, &mut map.A, n, 0, MatrixShape::N);
        }
    }

    // track the next sparse column to fill (assuming triu fill)
    let mut pcol = m + n; //next sparse column to fill
    let mut sparse_map_iter = map.sparse_maps.iter_mut();

    for (i, cone) in cones.iter().enumerate() {
        let row = cones.rng_cones[i].start + n;

        // add the Hs blocks in the lower right
        let blockdim = cone.numel();
        let block = &mut map.Hsblocks[cones.rng_blocks[i].clone()];

        if cone.Hs_is_diagonal() {
            K.fill_diag(block, row, blockdim);
        } else {
            K.fill_dense_triangle(block, row, blockdim, shape);
        }

        //add sparse expansions columns for sparse cones
        if cone.is_sparse_expandable() {
            let sc = cone.to_sparse_expansion().unwrap();
            let thismap = sparse_map_iter.next().unwrap();
            sc.csc_fill_sparsecone(thismap, K, row, pcol, shape);
            pcol += thismap.pdim();
        }
    }

    // backshift the colptrs to recover K.p again
    K.backshift_colptrs();

    // Now we can populate the index of the full diagonal.
    // We have filled in structural zeros on it everywhere.

    match shape {
        MatrixTriangle::Triu => {
            // matrix is triu, so diagonal is last in each column
            map.diag_full.copy_from_slice(&K.colptr[1..]);
            map.diag_full.iter_mut().for_each(|x| *x -= 1);
            // and the diagonal of just the upper left
            map.diagP.copy_from_slice(&K.colptr[1..=n]);
            map.diagP.iter_mut().for_each(|x| *x -= 1);
        }

        MatrixTriangle::Tril => {
            // matrix is tril, so diagonal is first in each column
            map.diag_full
                .copy_from_slice(&K.colptr[0..K.colptr.len() - 1]);
            // and the diagonal of just the upper left
            map.diagP.copy_from_slice(&K.colptr[0..n]);
        }
    }
}

#[test]
fn test_kkt_assembly_upper_lower() {
    let P = CscMatrix::from(&[
        [1., 2., 4.], //
        [0., 3., 5.], //
        [0., 0., 6.], //
    ]);
    let A = CscMatrix::from(&[
        [7., 0., 8.],  //
        [0., 9., 10.], //
        [1., 2., 3.],
    ]);

    let Ku_true_diag = CscMatrix::from(&[
        [1., 2., 4., 7., 0., 1.],  //
        [0., 3., 5., 0., 9., 2.],  //
        [0., 0., 6., 8., 10., 3.], //
        [0., 0., 0., -1., 0., 0.], //
        [0., 0., 0., 0., -1., 0.], //
        [0., 0., 0., 0., 0., -1.], //
    ]);

    let Kl_true_diag = CscMatrix::from(&[
        [1., 0., 0., 0., 0., 0.],   //
        [2., 3., 0., 0., 0., 0.],   //
        [4., 5., 6., 0., 0., 0.],   //
        [7., 0., 8., -1., 0., 0.],  //
        [0., 9., 10., 0., -1., 0.], //
        [1., 2., 3., 0., 0., -1.],  //
    ]);

    let Ku_true_dense = CscMatrix::from(&[
        [1., 2., 4., 7., 0., 1.],    //
        [0., 3., 5., 0., 9., 2.],    //
        [0., 0., 6., 8., 10., 3.],   //
        [0., 0., 0., -1., -1., -1.], //
        [0., 0., 0., 0., -1., -1.],  //
        [0., 0., 0., 0., 0., -1.],   //
    ]);

    let Kl_true_dense = CscMatrix::from(&[
        [1., 0., 0., 0., 0., 0.],    //
        [2., 3., 0., 0., 0., 0.],    //
        [4., 5., 6., 0., 0., 0.],    //
        [7., 0., 8., -1., 0., 0.],   //
        [0., 9., 10., -1., -1., 0.], //
        [1., 2., 3., -1., -1., -1.], //
    ]);

    // diagonal lower right block tests
    // --------------------------------
    let K = SupportedConeT::NonnegativeConeT(3);
    let cones = CompositeCone::new(&[K]);

    let (mut Ku, mapu) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Triu);
    for i in mapu.Hsblocks {
        Ku.nzval[i] = -1.;
    }
    assert_eq!(Ku, Ku_true_diag);

    let (mut Kl, mapl) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Tril);
    for i in mapl.Hsblocks {
        Kl.nzval[i] = -1.;
    }
    assert_eq!(Kl, Kl_true_diag);

    // dense lower right block tests
    // --------------------------------
    let K = SupportedConeT::ExponentialConeT();
    let cones = CompositeCone::new(&[K]);

    let (mut Ku, mapu) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Triu);
    for i in mapu.Hsblocks {
        Ku.nzval[i] = -1.;
    }
    assert_eq!(Ku, Ku_true_dense);

    let (mut Kl, mapl) = assemble_kkt_matrix(&P, &A, &cones, MatrixTriangle::Tril);
    for i in mapl.Hsblocks {
        Kl.nzval[i] = -1.;
    }
    assert_eq!(Kl, Kl_true_dense);
}
