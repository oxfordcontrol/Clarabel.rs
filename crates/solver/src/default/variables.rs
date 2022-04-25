use crate::conicvector::ConicVector;
use crate::cones::coneset::ConeSet;
use crate::algebra::*;

// ---------------
// Variables type for default problem format
// ---------------

pub struct DefaultVariables<T: FloatT = f64> {

    x: Vec<T>,
    s: ConicVector<T>,
    z: ConicVector<T>,
    τ: T,
    κ: T
}

impl<T: FloatT> DefaultVariables<T> {
    pub fn new(n: usize, cones: &ConeSet<T>) -> Self {

        let x = vec![T::zero(); n];
        let s = ConicVector::<T>::new(cones);
        let z = ConicVector::<T>::new(cones);
        let τ = T::one();
        let κ = T::one();


        Self {x,s,z,τ,κ}
    }
}
