// -----------------------------------
//  reverse the standard decomposition
// -----------------------------------

use crate::solver::chordal::ChordalInfo;
use crate::solver::DefaultVariables;
use crate::{
    algebra::*,
    solver::SupportedConeT::{self},
};
use std::iter::zip;

impl<T> ChordalInfo<T>
where
    T: FloatT,
{
    pub(crate) fn decomp_reverse_standard(
        &self,
        new_vars: &mut DefaultVariables<T>,
        old_vars: &DefaultVariables<T>,
        _old_cones: &[SupportedConeT<T>],
    ) {
        let H = &self.H.as_ref().unwrap();
        let (_, m) = new_vars.dims();

        H.gemv(&mut new_vars.s, &old_vars.s[m..], T::one(), T::zero());
        H.gemv(&mut new_vars.z, &old_vars.z[m..], T::one(), T::zero());

        //  to remove the overlaps we take the average of the values for
        //  each overlap by dividing by the number of blocks that overlap
        // in a particular entry, i.e. number of 1s in each row of H

        let (rows, nnzs) = number_of_overlaps_in_rows(H);

        for (ri, nnz) in zip(rows, nnzs) {
            new_vars.z[ri] /= nnz;
        }
    }
}

fn number_of_overlaps_in_rows<T>(A: &CscMatrix<T>) -> (Vec<usize>, Vec<T>)
where
    T: FloatT,
{
    // sum the entries row-wise
    let mut n_overlaps: Vec<T> = vec![T::zero(); A.m];
    A.row_sums(&mut n_overlaps);
    let ri = n_overlaps.iter().position_all(|&x| *x > T::one());

    //n_overlaps <- n_overlaps[ri]
    let n_overlaps: Vec<T> = ri.iter().map(|&i| n_overlaps[i]).collect();

    (ri, n_overlaps)
}
