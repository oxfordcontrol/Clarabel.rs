#![allow(non_snake_case)]

use super::datamap::*;
use crate::algebra::*;
use crate::solver::core::cones::CompositeCone;
use crate::solver::core::cones::*;
use num_traits::Zero;
use std::iter::zip;

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

pub fn assemble_kkt_matrix<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    shape: MatrixTriangle,
) -> (CscMatrix<T>, LDLDataMap) {
    let (m, n) = (A.nrows(), P.nrows());

    let n_socs = cones.type_count(SupportedConeTag::SecondOrderCone);
    let p_socs = 2 * n_socs;
    let n_genpows = cones.type_count(SupportedConeTag::GenPowerCone);
    let p_genpows = 3 * n_genpows;

    let mut maps = LDLDataMap::new(P, A, cones);

    // entries actually on the diagonal of P
    let nnz_diagP = P.count_diagonal_entries();

    // total entries in the Hs blocks
    let nnz_Hsblocks = maps.Hsblocks.len();

    // entries in the dense columns u/v of the
    // sparse SOC expansion terms.  2 is for
    // counting elements in both columns
    let nnz_SOC_vecs = 2 * maps.SOC_u.iter().fold(0, |acc, block| acc + block.len());

    // entries in the dense columns p.q.r of the
    // sparse generalized power expansion terms.
    let nnz_GenPow_vecs = maps.GenPow_p.iter().fold(0, |acc, block| acc + block.len())
        + maps.GenPow_q.iter().fold(0, |acc, block| acc + block.len())
        + maps.GenPow_r.iter().fold(0, |acc, block| acc + block.len());

    //entries in the sparse SOC diagonal extension block
    let nnz_SOC_ext = maps.SOC_D.len();
    //entries in the sparse generalized power diagonal extension block
    let nnz_GenPow_ext = maps.GenPow_D.len();

    let nnzKKT = P.nnz() +   // Number of elements in P
    n -                      // Number of elements in diagonal top left block
    nnz_diagP +              // remove double count on the diagonal if P has entries
    A.nnz() +                 // Number of nonzeros in A
    nnz_Hsblocks +         // Number of elements in diagonal below A'
    nnz_SOC_vecs +           // Number of elements in sparse SOC off diagonal columns
    nnz_SOC_ext  +          // Number of elements in diagonal of SOC extension
    nnz_GenPow_vecs +       // Number of elements in sparse generalized power off diagonal columns
    nnz_GenPow_ext; // Number of elements in diagonal of generalized power extension

    let Kdim = m + n + p_socs + p_genpows;
    let mut K = CscMatrix::<T>::spalloc((Kdim, Kdim), nnzKKT);

    _kkt_assemble_colcounts(&mut K, P, A, cones, (m, n, p_socs, p_genpows), shape);
    _kkt_assemble_fill(
        &mut K,
        &mut maps,
        P,
        A,
        cones,
        (m, n, p_socs, p_genpows),
        shape,
    );

    (K, maps)
}

fn _kkt_assemble_colcounts<T: FloatT>(
    K: &mut CscMatrix<T>,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    mnp: (usize, usize, usize, usize),
    shape: MatrixTriangle,
) {
    let (m, n, p_socs, p_genpows) = (mnp.0, mnp.1, mnp.2, mnp.3);

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

    // add the Hs blocks in the lower right
    for (i, cone) in cones.iter().enumerate() {
        let firstcol = cones.rng_cones[i].start + n;
        let blockdim = cone.numel();
        if cone.Hs_is_diagonal() {
            K.colcount_diag(firstcol, blockdim);
        } else {
            K.colcount_dense_triangle(firstcol, blockdim, shape);
        }
    }

    // count dense columns for each SOC
    let mut socidx = 0; // which SOC are we working on?

    for (i, cone) in cones.iter().enumerate() {
        if let SupportedCone::SecondOrderCone(SOC) = cone {
            // we will add the u and v columns for this cone
            let nvars = SOC.numel();
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
    K.colcount_diag(n + m, p_socs);

    // count dense columns for each generalized power cone
    let mut genpowidx = 0; // which Genpow are we working on?

    for (i, cone) in cones.iter().enumerate() {
        if let SupportedCone::GenPowerCone(GenPow) = cone {
            // we will add the p,q,r columns for this cone
            let nvars = GenPow.numel();
            let dim1 = GenPow.dim1;
            let dim2 = GenPow.dim2;
            let headidx = cones.rng_cones[i].start;

            // which column does q go into?
            let col = m + n + 2 * socidx + 3 * genpowidx;

            match shape {
                MatrixTriangle::Triu => {
                    K.colcount_colvec(dim1, headidx + n, col); // q column
                    K.colcount_colvec(dim2, headidx + n + dim1, col + 1); // r column
                    K.colcount_colvec(nvars, headidx + n, col + 2); // p column
                }
                MatrixTriangle::Tril => {
                    K.colcount_rowvec(dim1, col, headidx + n); // q row
                    K.colcount_rowvec(dim2, col + 1, headidx + n + dim1); // r row
                    K.colcount_rowvec(nvars, col + 2, headidx + n); // p row
                }
            }
            genpowidx = genpowidx + 1;
        }
    }

    // add diagonal block in the lower RH corner
    // to allow for the diagonal terms in generalized power expansion
    K.colcount_diag(n + m + p_socs, p_genpows);
}

fn _kkt_assemble_fill<T: FloatT>(
    K: &mut CscMatrix<T>,
    maps: &mut LDLDataMap,
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    cones: &CompositeCone<T>,
    mnp: (usize, usize, usize, usize),
    shape: MatrixTriangle,
) {
    let (m, n, p_socs, p_genpows) = (mnp.0, mnp.1, mnp.2, mnp.3);

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

    // add the the Hs blocks in the lower right
    for (i, (cone, rng_cone)) in zip(cones.iter(), &cones.rng_cones).enumerate() {
        let firstcol = rng_cone.start + n;
        let blockdim = cone.numel();
        let block = &mut maps.Hsblocks[cones.rng_blocks[i].clone()];
        if cone.Hs_is_diagonal() {
            K.fill_diag(block, firstcol, blockdim);
        } else {
            K.fill_dense_triangle(block, firstcol, blockdim, shape);
        }
    }

    // fill in dense columns for each SOC
    let mut socidx = 0; //which SOC are we working on?

    for (i, cone) in cones.iter().enumerate() {
        if let SupportedCone::SecondOrderCone(_) = cone {
            let headidx = cones.rng_cones[i].start;

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
    K.fill_diag(&mut maps.SOC_D, n + m, p_socs);

    // fill in dense columns for each generalized power cone
    let mut genpowidx = 0; //which generalized power cone are we working on?

    for (i, cone) in cones.iter().enumerate() {
        if let SupportedCone::GenPowerCone(cone) = cone {
            let headidx = cones.rng_cones[i].start;
            let dim1 = cone.dim1;

            // which column does q go into (if triu)?
            let col = m + n + 2 * socidx + 3 * genpowidx;

            // fill structural zeros for p,q,r columns for this cone
            match shape {
                MatrixTriangle::Triu => {
                    K.fill_colvec(&mut maps.GenPow_q[genpowidx], headidx + n, col); //q
                    K.fill_colvec(&mut maps.GenPow_r[genpowidx], headidx + n + dim1, col + 1); //r
                    K.fill_colvec(&mut maps.GenPow_p[genpowidx], headidx + n, col + 2);
                    //p
                    //v
                }
                MatrixTriangle::Tril => {
                    K.fill_rowvec(&mut maps.GenPow_q[genpowidx], col, headidx + n); //q
                    K.fill_rowvec(&mut maps.GenPow_r[genpowidx], col + 1, headidx + n + dim1); //r
                    K.fill_rowvec(&mut maps.GenPow_p[genpowidx], col + 2, headidx + n);
                    //p
                    //v
                }
            }

            genpowidx += 1;
        }
    }

    // fill in SOC diagonal extension with diagonal of structural zeros
    K.fill_diag(&mut maps.GenPow_D, n + m + p_socs, p_genpows);

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
