#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::SupportedConeT;

// ---------------
// Data type for default problem presolver
// ---------------

// PJG: updates required here
#[derive(Debug)]
pub(crate) struct PresolverRowReductionIndex {
    // vector of length = original RHS.   Entries are false
    // for those rows that should be eliminated before solve
    pub keep_logical: Vec<bool>,
}

/// Presolver data for the standard solver implementation

#[derive(Debug)]
pub(crate) struct Presolver<T> {
    // original cones of the problem
    // PJG: not currently used.  Here for future presolver
    pub(crate) _init_cones: Vec<SupportedConeT<T>>,

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
    /// create a new presolver object
    pub(crate) fn new(
        _A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        _settings: &DefaultSettings<T>,
    ) -> Self {
        let infbound = crate::get_infinity();

        // make copy of cones to protect from user interference
        let init_cones = cones.to_vec();
        let mfull = b.len();

        let (reduce_map, mreduced) = make_reduction_map(cones, b, infbound.as_T());

        Self {
            _init_cones: init_cones,
            reduce_map,
            mfull,
            mreduced,
            infbound,
        }
    }

    /// true if the presolver has reduced the problem
    pub(crate) fn is_reduced(&self) -> bool {
        self.reduce_map.is_some()
    }
    /// returns number of constraints eliminated
    pub(crate) fn count_reduced(&self) -> usize {
        self.mfull - self.mreduced
    }

    pub(crate) fn presolve(
        &self,
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
    ) -> (CscMatrix<T>, Vec<T>, Vec<SupportedConeT<T>>) {
        let (A_new, b_new) = self.reduce_A_b(A, b);
        let cones_new = self.reduce_cones(cones);

        (A_new, b_new, cones_new)
    }

    fn reduce_A_b(&self, A: &CscMatrix<T>, b: &[T]) -> (CscMatrix<T>, Vec<T>) {
        assert!(self.reduce_map.is_some());
        let map = self.reduce_map.as_ref().unwrap();

        let A = A.select_rows(&map.keep_logical);
        let b = b.select(&map.keep_logical);

        (A, b)
    }

    fn reduce_cones(&self, cones: &[SupportedConeT<T>]) -> Vec<SupportedConeT<T>> {
        assert!(self.reduce_map.is_some());
        let map = self.reduce_map.as_ref().unwrap();

        // assume that we will end up with the same
        // number of cones, despite small possibility
        // that some will be completely eliminated

        let mut cones_new = Vec::with_capacity(cones.len());
        let mut keep_iter = map.keep_logical.iter();

        for cone in cones {
            let numel_cone = cone.nvars();
            let markers = keep_iter.by_ref().take(numel_cone);

            if matches!(cone, SupportedConeT::NonnegativeConeT(_)) {
                let nkeep = markers.filter(|&b| *b).count();
                if nkeep > 0 {
                    cones_new.push(SupportedConeT::NonnegativeConeT(nkeep));
                }
            } else {
                //NB: take() is lazy, so must consume this block
                //to force keep_iter to advance to the next cone
                markers.last(); // skip this cone
                cones_new.push(cone.clone());
            }
        }

        cones_new
    }

    pub(crate) fn reverse_presolve(
        &self,
        solution: &mut DefaultSolution<T>,
        variables: &DefaultVariables<T>,
    ) {
        solution.x.copy_from(&variables.x);

        let map = self.reduce_map.as_ref().unwrap();
        let mut ctr = 0;

        for (idx, &keep) in map.keep_logical.iter().enumerate() {
            if keep {
                solution.s[idx] = variables.s[ctr];
                solution.z[idx] = variables.z[ctr];
                ctr += 1;
            } else {
                solution.s[idx] = self.infbound.as_T();
                solution.z[idx] = T::zero();
            }
        }
    }
}

fn make_reduction_map<T>(
    cones: &[SupportedConeT<T>],
    b: &[T],
    infbound: T,
) -> (Option<PresolverRowReductionIndex>, usize)
where
    T: FloatT,
{
    //assume we keep everything initially
    let mut keep_logical = vec![true; b.len()];
    let mut mreduced = b.len();

    // only try to reduce nn cones.  Make a slight contraction
    // so that we are firmly "less than" here
    let infbound = (T::one() - T::epsilon() * (10.).as_T()) * infbound;

    // we loop through b and remove any entries that are both infinite
    // and in a nonnegative cone

    let mut idx = 0; // index into the b vector

    for cone in cones {
        let numel_cone = cone.nvars();

        if matches!(cone, SupportedConeT::NonnegativeConeT(_)) {
            for _ in 0..numel_cone {
                if b[idx] > infbound {
                    keep_logical[idx] = false;
                    mreduced -= 1;
                }
                idx += 1;
            }
        } else {
            // skip this cone
            idx += numel_cone;
        }
    }

    let outoption = {
        if mreduced < b.len() {
            Some(PresolverRowReductionIndex { keep_logical })
        } else {
            None
        }
    };

    (outoption, mreduced)
}
