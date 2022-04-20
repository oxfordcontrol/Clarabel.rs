#![allow(non_snake_case)]
use crate::algebra::*;

// ---------------
// Residuals type for default problem format
// ---------------

pub struct DefaultResiduals<T: FloatT = f64> {

    // the main KKT residuals
    rx: Vec<T>,
    rz: Vec<T>,
    rτ: T,

    // partial residuals for infeasibility checks
    rx_inf: Vec<T>,
    rz_inf: Vec<T>,

    // various inner products.
    // NB: these are invariant w.r.t equilibration
    dot_qx: T,
    dot_bz: T,
    dot_sz: T,
    dot_xPx: T,

    // the product Px by itself. Required for infeasibilty checks
    Px:Vec<T>,
}

impl<T: FloatT> DefaultResiduals<T> {
    pub fn new(n: usize, m: usize) -> Self {

        let rx = vec![T::zero(); n];
        let rz = vec![T::zero(); m];
        let rτ = T::one();

        let rx_inf = vec![T::zero(); n];
        let rz_inf = vec![T::zero(); m];

        let Px = vec![T::zero(); n];


        Self {rx: rx, rz: rz, rτ: rτ, rx_inf: rx_inf, rz_inf: rz_inf,
            dot_qx:T::zero(), dot_bz:T::zero() , dot_sz:T::zero(), dot_xPx:T::zero(), Px: Px,}
    }
}
