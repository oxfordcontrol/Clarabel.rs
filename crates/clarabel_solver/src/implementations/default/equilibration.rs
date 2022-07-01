#![allow(non_snake_case)]
use clarabel_algebra::*;

// ---------------
// equilibration data
// ---------------

pub struct DefaultEquilibration<T: FloatT = f64> {
    // scaling matrices for problem data equilibration
    // fields d,e,dinv,einv are vectors of scaling values
    // to be treated as diagonal scaling data
    pub d: Vec<T>,
    pub dinv: Vec<T>,
    pub e: Vec<T>,
    pub einv: Vec<T>,

    // overall scaling for objective function
    pub c: T,
}

impl<T: FloatT> DefaultEquilibration<T> {
    pub fn new(n: usize, m: usize) -> Self {
        // Left/Right diagonal scaling for problem data
        let d = vec![T::one(); n];
        let dinv = vec![T::one(); n];
        let e = vec![T::one(); m];
        let einv = vec![T::one(); m];

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
