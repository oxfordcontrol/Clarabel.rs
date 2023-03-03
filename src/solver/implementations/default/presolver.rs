#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::get_infinity;
use crate::solver::SupportedConeT;

// ---------------
// Data type for default problem presolver
// ---------------

pub struct Presolver<T> {
    // possibly reduced internal copy of user cone specification
    pub cone_specs: Vec<SupportedConeT<T>>,

    // vector of length = original RHS.   Entries are false
    // for those rows that should be eliminated before solve
    pub reduce_idx: Option<Vec<bool>>,

    // vector of length = reduced RHS, taking values
    // that map reduced b back to their original index
    lift_map: Option<Vec<usize>>,

    // size of original and reduced RHS, respectively
    mfull: usize,
    mreduced: usize,
}

impl<T> Presolver<T>
where
    T: FloatT,
{
    pub fn new(
        _A: &CscMatrix<T>,
        b: &[T],
        cone_specs: &[SupportedConeT<T>],
        settings: &DefaultSettings<T>,
    ) -> Self {
        // first make copy of cone_specs to protect from user interference
        let mut cone_specs = cone_specs.to_vec();
        let mfull = b.len();

        let (reduce_idx, lift_map, mreduced) = {
            if !settings.presolve_enable {
                (None, None, mfull)
            } else {
                let (reduce_idx, lift_map) = reduce_cones(&mut cone_specs, b);
                match lift_map {
                    None => (None, None, mfull),
                    Some(lm) => {
                        let mreduced = lm.len();
                        (reduce_idx, Some(lm), mreduced)
                    }
                }
            }
        };

        Self {
            cone_specs,
            reduce_idx,
            lift_map,
            mfull,
            mreduced,
        }
    }

    pub fn is_reduced(&self) -> bool {
        self.mfull != self.mreduced
    }
    pub fn count_reduced(&self) -> usize {
        self.mfull - self.mreduced
    }
}

fn reduce_cones<T>(
    cone_specs: &mut [SupportedConeT<T>],
    b: &[T],
) -> (Option<Vec<bool>>, Option<Vec<usize>>)
where
    T: FloatT,
{
    let mut reduce_idx = vec![true; b.len()];

    // we loop through the finite_idx and shrink any nonnegative
    // cones that are marked as having infinite right hand sides.
    // Mark the corresponding entries as zero in the reduction index

    let mut is_reduced = false;
    let mut bptr = 0; // index into the b vector

    for cone in cone_specs.iter_mut() {
        let numel_cone = cone.nvars();

        // only try to reduce nn cones
        let infbound = (T::one() - T::epsilon()) * get_infinity().as_T();
        if matches!(cone, SupportedConeT::NonnegativeConeT(_)) {
            let mut num_finite = 0;
            for i in bptr..(bptr + numel_cone) {
                if b[i] < infbound {
                    num_finite += 1
                } else {
                    reduce_idx[i] = false
                }
            }
            if num_finite < numel_cone {
                // contract the cone to a smaller size
                *cone = SupportedConeT::NonnegativeConeT(num_finite);
                is_reduced = true
            }
        }

        bptr += numel_cone;
    }

    // if we reduced anything then return the reduce_idx and a
    // make of the entries to keep back into the original vector

    if is_reduced {
        let lift_map = findall(&reduce_idx);
        (Some(reduce_idx), Some(lift_map))
    } else {
        (None, None)
    }
}

fn findall(reduce_idx: &[bool]) -> Vec<usize> {
    let map = reduce_idx
        .iter()
        .enumerate()
        .filter(|(_, &r)| r)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    map
}
