#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::cones::*;

pub struct LDLDataMap {
    pub P: Vec<usize>,
    pub A: Vec<usize>,
    pub WtWblocks: Vec<Vec<usize>>, //indices of the lower RHS blocks (by cone)
    pub SOC_u: Vec<Vec<usize>>,     //off diag dense columns u
    pub SOC_v: Vec<Vec<usize>>,     //off diag dense columns v
    pub SOC_D: Vec<usize>,          //diag of just the sparse SOC expansion D

    // all of above terms should be disjoint and their union
    // should cover all of the user data in the KKT matrix.  Now
    // we make two last redundant indices that will tell us where
    // the whole diagonal is, including structural zeros.
    pub diagP: Vec<usize>,
    pub diag_full: Vec<usize>,
}

impl LDLDataMap {
    pub fn new<T: FloatT>(Pmat: &CscMatrix<T>, Amat: &CscMatrix<T>, cones: &ConeSet<T>) -> Self {
        let (m, n) = (Amat.nrows(), Pmat.nrows());
        let P = vec![0; Pmat.nnz()];
        let A = vec![0; Amat.nnz()];

        // the diagonal of the ULHS KKT block P.
        // NB : we fill in structural zeros here even if the matrix
        // P is empty (e.g. as in an LP), so we can have entries in
        // index Pdiag that are not present in the index P
        let diagP = vec![0; n];

        // make an index for each of the WtW blocks for each cone
        let WtWblocks = _allocate_kkt_WtW_blocks::<T, usize>(cones);

        // now do the SOC expansion pieces
        let nsoc = cones.type_counts[&SupportedCones::SecondOrderConeT];
        let p = 2 * nsoc;
        let SOC_D = vec![0; p];

        let mut SOC_u = Vec::<Vec<usize>>::with_capacity(nsoc);
        let mut SOC_v = Vec::<Vec<usize>>::with_capacity(nsoc);

        for (i, cone) in cones.iter().enumerate() {
            if cones.types[i] == SupportedCones::SecondOrderConeT {
                SOC_u.push(vec![0; cone.numel()]);
                SOC_v.push(vec![0; cone.numel()]);
            }
        }

        let diag_full = vec![0, m + n + p];

        Self {
            P,
            A,
            WtWblocks,
            SOC_u,
            SOC_v,
            SOC_D,
            diagP,
            diag_full,
        }
    }
}
