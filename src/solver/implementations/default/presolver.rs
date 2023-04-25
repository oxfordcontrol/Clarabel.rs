#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::SupportedConeT;

// ---------------
// Data type for default problem presolver
// ---------------

#[derive(Debug)]
pub(crate) struct PresolverRowReductionIndex {
    // vector of length = original RHS.   Entries are false
    // for those rows that should be eliminated before solve
    pub keep_logical: Vec<bool>,

    // vector of length = reduced RHS, taking values
    // that map reduced b back to their original index
    // This is just findall(keep_logical) and is held for
    // efficient solution repopulation
    pub keep_index: Vec<usize>,
}

/// Presolver data for the standard solver implementation

#[derive(Debug)]
pub struct Presolver<T> {
    // possibly reduced internal copy of user cone specification
    pub(crate) cone_specs: Vec<SupportedConeT<T>>,

    //record of reduced constraints for NN cones with inf bounds
    pub(crate) reduce_map: Option<PresolverRowReductionIndex>,

    // size of original and reduced RHS, respectively
    pub(crate) mfull: usize,
    pub(crate) mreduced: usize,

    // inf bound that was taken from the module level
    // and should be applied throughout.   Held here so
    // that any subsequent change to the module's state
    // won't mess up our solver mid-solve
    pub(crate) infbound: f64,
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
        let infbound = crate::solver::get_infinity();

        // make copy of cone_specs to protect from user interference
        let mut cone_specs = cone_specs.to_vec();
        let mfull = b.len();

        let (reduce_map, mreduced) = {
            if settings.presolve_enable {
                reduce_cones(&mut cone_specs, b, infbound.as_T())
            } else {
                (None, mfull)
            }
        };

        Self {
            cone_specs,
            reduce_map,
            mfull,
            mreduced,
            infbound,
        }
    }

    pub fn is_reduced(&self) -> bool {
        self.reduce_map.is_some()
    }
    pub fn count_reduced(&self) -> usize {
        self.mfull - self.mreduced
    }
}

fn reduce_cones<T>(
    cone_specs: &mut [SupportedConeT<T>],
    b: &[T],
    infbound: T,
) -> (Option<PresolverRowReductionIndex>, usize)
where
    T: FloatT,
{
    //assume we keep everything initially
    let mut keep_logical = vec![true; b.len()];
    let mut mreduced = b.len();

    // we loop through b and remove any entries that are both infinite
    // and in a nonnegative cone

    let mut is_reduced = false;
    let mut bptr = 0; // index into the b vector

    for cone in cone_specs.iter_mut() {
        let numel_cone = cone.nvars();

        // only try to reduce nn cones.  Make a slight contraction
        // so that we are firmly "less than" here
        let infbound = (T::one() - T::epsilon() * (10.).as_T()) * infbound;

        if matches!(cone, SupportedConeT::NonnegativeConeT(_)) {
            let mut num_finite = 0;
            for i in bptr..(bptr + numel_cone) {
                if b[i] < infbound {
                    num_finite += 1;
                } else {
                    keep_logical[i] = false;
                    mreduced -= 1;
                }
            }
            if num_finite < numel_cone {
                // contract the cone to a smaller size
                *cone = SupportedConeT::NonnegativeConeT(num_finite);
                is_reduced = true;
            }
        }

        bptr += numel_cone;
    }

    let outoption = {
        if is_reduced {
            let keep_index = findall(&keep_logical);
            Some(PresolverRowReductionIndex {
                keep_logical,
                keep_index,
            })
        } else {
            None
        }
    };

    (outoption, mreduced)
}

fn findall(keep_logical: &[bool]) -> Vec<usize> {
    let map = keep_logical
        .iter()
        .enumerate()
        .filter(|(_, &r)| r)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    map
}
