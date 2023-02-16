use super::*;
use crate::algebra::*;
use crate::solver::core::cones::PrimalOrDualCone;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    traits::{Settings, Variables},
    ScalingStrategy,
};

// ---------------
// Variables type for default problem format
// ---------------

/// Standard-form solver type implementing the [Variables](crate::solver::core::traits::Variables) trait
#[derive(Debug)]
pub struct DefaultVariables<T> {
    /// scaled primal variables
    pub x: Vec<T>,
    /// slack variables
    pub s: Vec<T>,
    /// scaled dual variables
    pub z: Vec<T>,
    /// homogenization scalar τ
    pub τ: T,
    /// homogenization scalar κ
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
    type SE = DefaultSettings<T>;

    fn calc_mu(&mut self, residuals: &DefaultResiduals<T>, cones: &CompositeCone<T>) -> T {
        let denom = T::from(cones.degree() + 1).unwrap();
        (residuals.dot_sz + self.τ * self.κ) / denom
    }

    fn affine_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &CompositeCone<T>,
    ) {
        self.x.copy_from(&residuals.rx);
        self.z.copy_from(&residuals.rz);
        cones.affine_ds(&mut self.s, &variables.s);
        self.τ = residuals.rτ;
        self.κ = variables.τ * variables.κ;
    }

    fn combined_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &mut CompositeCone<T>,
        step: &mut Self,
        σ: T,
        μ: T,
    ) {
        let dotσμ = σ * μ;
        self.x.axpby(T::one() - σ, &residuals.rx, T::zero()); //self.x  = (1 - σ)*rx
        self.τ = (T::one() - σ) * residuals.rτ;
        self.κ = -dotσμ + step.τ * step.κ + variables.τ * variables.κ;

        // ds is different for symmetric and asymmetric cones:
        // Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
        // Asymmetric cones: d.s = s + σμ*g(z)
        cones.combined_ds_shift(&mut self.z, &mut step.z, &mut step.s, dotσμ);

        //We are relying on d.s = affine_ds already here
        self.s.axpby(T::one(), &self.z, T::one());

        // now we copy the scaled res for rz and d.z is no longer work
        self.z.axpby(T::one() - σ, &residuals.rz, T::zero());
    }

    fn calc_step_length(
        &self,
        step: &Self,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
        steptype: &'static str,
    ) -> T {
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

        let α = [ατ, ακ, T::one()].minimum();
        let (αz, αs) = cones.step_length(&step.z, &step.s, &self.z, &self.s, settings.core(), α);

        // itself only allows for a single maximum value.
        // To enable split lengths, we need to also pass a
        // tuple of limits to the step_length function of
        // every cone
        let mut α = T::min(αz, αs);

        if steptype == "combined" {
            α *= settings.core().max_step_fraction
        }

        α
    }

    fn add_step(&mut self, step: &Self, α: T) {
        self.x.axpby(α, &step.x, T::one());
        self.s.axpby(α, &step.s, T::one());
        self.z.axpby(α, &step.z, T::one());
        self.τ += α * step.τ;
        self.κ += α * step.κ;
    }

    fn symmetric_initialization(
        &mut self,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) {
        let min_margin = T::one() - settings.max_step_fraction;

        _shift_to_cone_interior(&mut self.s, cones, min_margin, PrimalOrDualCone::PrimalCone);
        _shift_to_cone_interior(&mut self.z, cones, min_margin, PrimalOrDualCone::DualCone);

        self.τ = T::one();
        self.κ = T::one();
    }

    fn unit_initialization(&mut self, cones: &CompositeCone<T>) {
        cones.unit_initialization(&mut self.z, &mut self.s);

        self.x.set(T::zero());
        self.τ = T::one();
        self.κ = T::one();
    }

    fn copy_from(&mut self, src: &Self) {
        self.x.copy_from(&src.x);
        self.s.copy_from(&src.s);
        self.z.copy_from(&src.z);
        self.τ = src.τ;
        self.κ = src.κ;
    }

    fn scale_cones(
        &self,
        cones: &mut CompositeCone<T>,
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        cones.update_scaling(&self.s, &self.z, μ, scaling_strategy)
    }

    fn barrier(&self, step: &Self, α: T, cones: &CompositeCone<T>) -> T {
        let central_coef = (cones.degree() + 1).as_T();

        let cur_τ = self.τ + α * step.τ;
        let cur_κ = self.κ + α * step.κ;

        // compute current μ
        let sz = <[T] as VectorMath<T>>::dot_shifted(&self.z, &self.s, &step.z, &step.s, α);
        let μ = (sz + cur_τ * cur_κ) / central_coef;

        // barrier terms from gap and scalars
        let mut barrier = central_coef * μ.logsafe() - cur_τ.logsafe() - cur_κ.logsafe();

        // barriers from the cones
        let (z, s) = (&self.z, &self.s);
        let (dz, ds) = (&step.z, &step.s);

        barrier += cones.compute_barrier(z, s, dz, ds, α);

        barrier
    }

    fn rescale(&mut self) {
        let scale = T::max(self.τ, self.κ);
        let invscale = scale.recip();

        self.x.scale(invscale);
        self.z.scale(invscale);
        self.s.scale(invscale);
        self.τ *= invscale;
        self.κ *= invscale;
    }
}

fn _sum_pos<T>(z: &[T]) -> T
where
    T: FloatT,
{
    let mut out = T::zero();
    for &zi in z.iter() {
        out += match zi > T::zero() {
            true => zi,
            false => T::one(),
        };
    }
    out
}

fn _shift_to_cone_interior<T>(
    z: &mut [T],
    cones: &CompositeCone<T>,
    min_margin: T,
    pd: PrimalOrDualCone,
) where
    T: FloatT,
{
    let margin = cones.unit_margin(z, pd);
    let nhood = _sum_pos(z) / (z.len()).as_T();
    let nhood = T::max(T::one(), nhood) * (0.1).as_T();

    if margin <= T::zero() {
        //done in two stages since otherwise (1-α) = -α for
        //large α, which makes z exactly 0. (or worse, -0.0 )
        cones.scaled_unit_shift(z, -margin, pd);
        cones.scaled_unit_shift(z, T::max(T::one(), nhood), pd);
    } else if margin < min_margin {
        // margin is positive but small.
        cones.scaled_unit_shift(z, nhood, pd);
    } else {
        // good margin, but still shift explicitly by
        // zero to catch any elements in the zero cone
        // that need to be forced to zero
        cones.scaled_unit_shift(z, T::zero(), pd);
    }
}
