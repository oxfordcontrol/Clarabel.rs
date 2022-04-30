use crate::algebra::*;
use crate::cones::coneset::ConeSet;
use crate::conicvector::ConicVector;
use crate::default::*;

// ---------------
// Variables type for default problem format
// ---------------

pub struct DefaultVariables<T: FloatT = f64> {
    x: Vec<T>,
    s: ConicVector<T>,
    z: ConicVector<T>,
    τ: T,
    κ: T,
}

impl<T: FloatT> DefaultVariables<T> {
    pub fn new(n: usize, cones: &ConeSet<T>) -> Self {
        let x = vec![T::zero(); n];
        let s = ConicVector::<T>::new(cones);
        let z = ConicVector::<T>::new(cones);
        let τ = T::one();
        let κ = T::one();

        Self { x, s, z, τ, κ }
    }
}

impl<T:FloatT> Variables<T> for DefaultVariables<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type R = DefaultResiduals<T>;
    type C = ConeSet<T>;

    fn calc_mu(&mut self, residuals: &DefaultResiduals<T>, cones: &ConeSet<T>) -> T
    {
        todo!();
    }

    fn calc_affine_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        data: &DefaultProblemData<T>,
        variables: &Self,
        cones: &ConeSet<T>,
    )
    {
        todo!();
    }

    fn calc_combined_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        data: &DefaultProblemData<T>,
        variables: &Self,
        cones: &ConeSet<T>,
        step_lhs: &Self,
        σ: T,
        μ: T,
    )
    {
        todo!();
    }

    fn calc_step_length(&mut self, step_lhs: &Self, cones: &ConeSet<T>) -> T
    {
        todo!();
    }

    fn add_step(&mut self, step_lhs: &Self, α: T)
    {
        todo!();
    }

    fn shift_to_cone(&mut self, cones: &ConeSet<T>)
    {
        todo!();
    }

    fn scale_cones(&self, cones: &mut ConeSet<T>)
    {
        todo!();
    }

}
