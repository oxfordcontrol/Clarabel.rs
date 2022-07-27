#![allow(non_snake_case)]

use super::datamap::*;
use crate::algebra::*;
use crate::solver::core::cones::{CompositeCone, SupportedCones};
use num_traits::Zero;

pub(crate) fn allocate_kkt_WtW_blocks<T, Z>(cones: &CompositeCone<T>) -> Vec<Z>
where
    T: FloatT,
    Z: Zero + Clone,
{
    let mut nnz = 0;
    if let Some(rng_last) = cones.rng_cones.last() {
        nnz = (*rng_last).end;
    }
    vec![Z::zero(); nnz]
}

pub fn assemble_kkt_matrix<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    shape: MatrixTriangle,
) -> (CscMatrix<T>, LDLDataMap) {
    let (m, n) = (A.nrows(), P.nrows());
    let n_socs = cones.type_count("SecondOrderConeT");
    let p = 2 * n_socs;

    let mut maps = LDLDataMap::new(P, A, cones);

    // entries actually on the diagonal of P
    let nnz_diagP = P.count_diagonal_entries();

    // total entries in the WtW blocks
    let nnz_WtW_blocks = maps.WtWblocks.len();

    // entries in the dense columns u/v of the
    // sparse SOC expansion terms.  2 is for
    // counting elements in both columns
    let nnz_SOC_vecs = 2 * maps.SOC_u.iter().fold(0, |acc, block| acc + block.len());

    //entries in the sparse SOC diagonal extension block
    let nnz_SOC_ext = maps.SOC_D.len();

    let nnzKKT = P.nnz() +   // Number of elements in P
    n -                      // Number of elements in diagonal top left block
    nnz_diagP +              // remove double count on the diagonal if P has entries
    A.nnz() +                 // Number of nonzeros in A
    nnz_WtW_blocks +         // Number of elements in diagonal below A'
    nnz_SOC_vecs +           // Number of elements in sparse SOC off diagonal columns
    nnz_SOC_ext; // Number of elements in diagonal of SOC extension

    let mut K = CscMatrix::<T>::spalloc(m + n + p, m + n + p, nnzKKT);

    _kkt_assemble_colcounts(&mut K, P, A, cones, (m, n, p), shape);
    _kkt_assemble_fill(&mut K, &mut maps, P, A, cones, (m, n, p), shape);

    (K, maps)
}

fn _kkt_assemble_colcounts<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    mnp: (usize, usize, usize),
    shape: MatrixTriangle,
) {
    let (m, n, p) = (mnp.0, mnp.1, mnp.2);

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

    // add the the WtW blocks in the lower right
    for (i, cone) in cones.iter().enumerate() {
        let firstcol = cones.rng_cones[i].start + n;
        let blockdim = cone.numel();
        if cone.WtW_is_diagonal() {
            K.colcount_diag(firstcol, blockdim);
        } else {
            K.colcount_dense_triangle(firstcol, blockdim, shape);
        }
    }

    // count dense columns for each SOC
    let mut socidx = 0; // which SOC are we working on?

    for (i, cone) in cones.iter().enumerate() {
        if matches!(cones.types[i], SupportedCones::SecondOrderConeT(_)) {
            // we will add the u and v columns for this cone
            let nvars = cone.numel();
            let headidx = cones.rng_cones[i].start;

            // which column does u go into?
            let col = m + n + 2 * socidx;

            match shape {
                MatrixTriangle::Triu => {
                    K.colcount_colvec(nvars, headidx + n, col); // u column
                    K.colcount_colvec(nvars, headidx + n, col + 1); // v column
                }
                MatrixTriangle::Tril => {
                    K.colcount_rowvec(nvars, col, headidx + n); // u row
                    K.colcount_rowvec(nvars, col + 1, headidx + n); // v row
                }
            }
            socidx += 1;
        }
    }

    // add diagonal block in the lower RH corner
    // to allow for the diagonal terms in SOC expansion
    K.colcount_diag(n + m, p);
}

fn _kkt_assemble_fill<T: FloatT>(
    K: &mut CscMatrix<T>,
    maps: &mut LDLDataMap,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    mnp: (usize, usize, usize),
    shape: MatrixTriangle,
) {
    let (m, n, p) = (mnp.0, mnp.1, mnp.2);

    // cumsum total entries to convert to K.p
    K.colcount_to_colptr();

    match shape {
        MatrixTriangle::Triu => {
            K.fill_block(P, &mut maps.P, 0, 0, MatrixShape::N);
            K.fill_missing_diag(P, 0); // after adding P, since triu form
                                       // fill in value for A, top right (transposed/rowwise)
            K.fill_block(A, &mut maps.A, 0, n, MatrixShape::T);
        }
        MatrixTriangle::Tril => {
            K.fill_missing_diag(P, 0); // before adding P, since tril form
            K.fill_block(P, &mut maps.P, 0, 0, MatrixShape::T);
            // fill in value for A, bottom left (not transposed)
            K.fill_block(A, &mut maps.A, n, 0, MatrixShape::N);
        }
    }

    // add the the WtW blocks in the lower right
    for (i, (cone, rng_cone)) in cones.iter().zip(cones.rng_cones.iter()).enumerate() {
        let firstcol = rng_cone.start + n;
        let blockdim = cone.numel();
        let block = &mut maps.WtWblocks[cones.rng_blocks[i].clone()];
        if cone.WtW_is_diagonal() {
            K.fill_diag(block, firstcol, blockdim);
        } else {
            K.fill_dense_triangle(block, firstcol, blockdim, shape);
        }
    }

    // fill in dense columns for each SOC
    let mut socidx = 0; //which SOC are we working on?

    for (i, rng) in cones.rng_cones.iter().enumerate() {
        if matches!(cones.types[i], SupportedCones::SecondOrderConeT(_)) {
            let headidx = rng.start;

            // which column does u go into (if triu)?
            let col = m + n + 2 * socidx;

            // fill structural zeros for u and v columns for this cone
            // note v is the first extra row/column, u is second
            match shape {
                MatrixTriangle::Triu => {
                    K.fill_colvec(&mut maps.SOC_v[socidx], headidx + n, col); //u
                    K.fill_colvec(&mut maps.SOC_u[socidx], headidx + n, col + 1);
                    //v
                }
                MatrixTriangle::Tril => {
                    K.fill_rowvec(&mut maps.SOC_v[socidx], col, headidx + n); //u
                    K.fill_rowvec(&mut maps.SOC_u[socidx], col + 1, headidx + n);
                    //v
                }
            }

            socidx += 1;
        }
    }

    // fill in SOC diagonal extension with diagonal of structural zeros
    K.fill_diag(&mut maps.SOC_D, n + m, p);

    // backshift the colptrs to recover K.p again
    K.backshift_colptrs();

    // Now we can populate the index of the full diagonal.
    // We have filled in structural zeros on it everywhere.

    match shape {
        MatrixTriangle::Triu => {
            // matrix is triu, so diagonal is last in each column
            maps.diag_full.copy_from_slice(&K.colptr[1..]);
            maps.diag_full.iter_mut().for_each(|x| *x -= 1);
            // and the diagonal of just the upper left
            maps.diagP.copy_from_slice(&K.colptr[1..=n]);
            maps.diagP.iter_mut().for_each(|x| *x -= 1);
        }

        MatrixTriangle::Tril => {
            // matrix is tril, so diagonal is first in each column
            maps.diag_full
                .copy_from_slice(&K.colptr[0..K.colptr.len() - 1]);
            // and the diagonal of just the upper left
            maps.diagP.copy_from_slice(&K.colptr[0..n]);
        }
    }
}
