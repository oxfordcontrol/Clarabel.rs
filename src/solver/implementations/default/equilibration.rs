#![allow(non_snake_case)]
use crate::algebra::*;

// ---------------
// equilibration data
// ---------------

/// Data from the Ruiz equilibration procedure
pub struct DefaultEquilibrationData<T> {
    // scaling matrices for problem data equilibration
    // fields d,e,dinv,einv are vectors of scaling values
    // to be treated as diagonal scaling data
    /// Vector of variable scaling terms
    pub d: Vec<T>,
    /// Vector of inverse variable scaling terms
    pub dinv: Vec<T>,
    /// Vector of constraint scaling terms
    pub e: Vec<T>,
    /// Vector of inverse constraint scaling terms
    pub einv: Vec<T>,
    /// overall scaling for objective function
    pub c: T,
}

impl<T> DefaultEquilibrationData<T>
where
    T: FloatT,
{
    /// creates a new equilibration object
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
