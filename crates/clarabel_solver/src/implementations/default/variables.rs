use super::*;
use crate::core::{
    cones::{CompositeCone, Cone},
    traits::Variables,
};
use clarabel_algebra::*;

// ---------------
// Variables type for default problem format
// ---------------

pub struct DefaultVariables<T> {
    pub x: Vec<T>,
    pub s: Vec<T>,
    pub z: Vec<T>,
    pub τ: T,
    pub κ: T,
}

impl<T> DefaultVariables<T>
where
    T: FloatT,
{
    pub fn new(n: usize, m: usize) -> Self {
        let x = vec![T::zero(); n];
        let s = vec![T::zero(); m];
        let z = vec![T::zero(); m];
        let τ = T::one();
        let κ = T::one();

        Self { x, s, z, τ, κ }
    }
}

impl<T> Variables<T> for DefaultVariables<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type R = DefaultResiduals<T>;
    type C = CompositeCone<T>;

    fn calc_mu(&mut self, residuals: &DefaultResiduals<T>, cones: &CompositeCone<T>) -> T {
        let denom = T::from(cones.degree() + 1).unwrap();
        (residuals.dot_sz + self.τ * self.κ) / denom
    }

    fn calc_affine_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &CompositeCone<T>,
    ) {
        self.x.copy_from(&residuals.rx);
        self.z.copy_from(&residuals.rz);
        cones.λ_circ_λ(&mut self.s);
        self.τ = residuals.rτ;
        self.κ = variables.τ * variables.κ;
    }

    fn calc_combined_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &CompositeCone<T>,
        step: &mut Self,
        σ: T,
        μ: T,
    ) {
        self.x.axpby(T::one() - σ, &residuals.rx, T::zero()); //self.x  = (1 - σ)*rx
        self.τ = (T::one() - σ) * residuals.rτ;
        self.κ = -σ * μ + step.τ * step.κ + variables.τ * variables.κ;

        //self.s must be assembled carefully if we want to be economical with
        //allocated memory.  Will modify the step.z and step.s in place since
        //they are from the affine step and not needed anymore.

        //Will also use d.z as a temporary work vector here. Note that we don't
        //want to have aliasing vector arguments to gemv_W or gemv_Winv, so we
        //need to copy into a temporary variable to assign #Δz = WΔz and Δs = W⁻¹Δs

        let tmp = &mut self.z; //alias
        tmp.copy_from(&step.z); //copy for safe call to gemv_W
        cones.gemv_W(MatrixShape::N, tmp, &mut step.z, T::one(), T::zero()); //Δz <- WΔz
        tmp.copy_from(&step.s); //copy for safe call to gemv_Winv
        cones.gemv_Winv(MatrixShape::T, tmp, &mut step.s, T::one(), T::zero()); //Δs <- W⁻¹Δs
        cones.circ_op(tmp, &step.s, &step.z); //tmp = W⁻¹Δs ∘ WΔz
        cones.add_scaled_e(tmp, -σ * μ); //tmp = W⁻¹Δs ∘ WΔz - σμe

        //We are relying on d.s = λ ◦ λ already from the affine step here
        self.s.axpby(T::one(), &self.z, T::one());

        // now we copy the scaled res for rz and d.z is no longer work
        self.z.axpby(T::one() - σ, &residuals.rz, T::zero());
    }

    fn calc_step_length(&mut self, step: &Self, cones: &CompositeCone<T>) -> T {
        let ατ = {
            if step.τ < T::zero() {
                -self.τ / step.τ
            } else {
                T::max_value()
            }
        };

        let ακ = {
            if step.κ < T::zero() {
                -self.κ / step.κ
            } else {
                T::max_value()
            }
        };

        let (αz, αs) = cones.step_length(&step.z, &step.s, &self.z, &self.s);

        vec![ατ, ακ, αz, αs, T::one()].minimum()
    }

    fn add_step(&mut self, step: &Self, α: T) {
        self.x.axpby(α, &step.x, T::one());
        self.s.axpby(α, &step.s, T::one());
        self.z.axpby(α, &step.z, T::one());
        self.τ += α * step.τ;
        self.κ += α * step.κ;
    }

    fn shift_to_cone(&mut self, cones: &CompositeCone<T>) {
        cones.shift_to_cone(&mut self.s);
        cones.shift_to_cone(&mut self.z);

        self.τ = T::one();
        self.κ = T::one();
    }

    fn scale_cones(&self, cones: &mut CompositeCone<T>) {
        cones.update_scaling(&self.s, &self.z);
    }
}
