#![allow(non_snake_case)]
use crate::algebra::*;
use crate::conicvector::ConicVector;
use crate::ConeSet;

// ---------------
// equilibration data
// ---------------

pub struct DefaultEquilibration<T: FloatT = f64> {
    // scaling matrices for problem data equilibration
    // fields d,e,dinv,einv are vectors of scaling values
    // to be treated as diagonal scaling data
    pub d: Vec<T>,
    pub dinv: Vec<T>,
    pub e: ConicVector<T>,
    pub einv: ConicVector<T>,

    // overall scaling for objective function
    pub c: T,
}

impl<T: FloatT> DefaultEquilibration<T> {
    pub fn new(nvars: usize, cones: &ConeSet<T>) -> Self {
        // Left/Right diagonal scaling for problem data
        let d = vec![T::zero(); nvars];
        let dinv = vec![T::zero(); nvars];

        // PJG : note that this double initializes
        // e / einv because the ConicVector constructor
        // first initializes to zero.   Could be improved.
        let mut e = ConicVector::<T>::new(cones);
        e.fill(T::one());
        let mut einv = ConicVector::<T>::new(cones);
        einv.fill(T::one());

        let c = T::one();

        Self {
            d,
            dinv,
            e,
            einv,
            c,
        }
    }
}
