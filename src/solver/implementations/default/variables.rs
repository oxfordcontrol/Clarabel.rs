use super::*;
use crate::algebra::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone, PrimalOrDualCone},
    traits::{Settings, Variables},
    ScalingStrategy, StepDirection,
};

// ---------------
// Variables type for default problem format
// ---------------

/// Standard-form solver type implementing the [`Variables`](crate::solver::core::traits::Variables) trait
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

impl<T: std::fmt::Display + std::fmt::Debug> std::fmt::Debug for DefaultVariables<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "x: {:?}\ns: {:?}\nz: {:?}\nτ: {:?}\nκ: {:?}\n",
            self.x, self.s, self.z, self.τ, self.κ
        )
    }
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
        m: T,
    ) {
        let dotσμ = σ * μ;

        self.x.axpby(T::one() - σ, &residuals.rx, T::zero()); //self.x  = (1 - σ)*rx
        self.τ = (T::one() - σ) * residuals.rτ;
        self.κ = -dotσμ + m * step.τ * step.κ + variables.τ * variables.κ;

        // ds is different for symmetric and asymmetric cones:
        // Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
        // Asymmetric cones: d.s = s + σμ*g(z)

        // we want to scale the Mehotra correction in the symmetric
        // case by M, so just scale step_z by M.  This is an unnecessary
        // vector operation (since it amounts to M*z'*s), but it
        // doesn't happen very often
        if m != T::one() {
            step.z.scale(m);
        }

        cones.combined_ds_shift(&mut self.z, &mut step.z, &mut step.s, dotσμ);

        //We are relying on d.s = affine_ds already here
        self.s.axpby(T::one(), &self.z, T::one());

        // now we copy the scaled res for rz and d.z is no longer work
        self.z.axpby(T::one() - σ, &residuals.rz, T::zero());
    }

    fn calc_step_length(
        &self,
        step: &Self,
        cones: &mut CompositeCone<T>,
        settings: &DefaultSettings<T>,
        step_direction: StepDirection,
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

        if step_direction == StepDirection::Combined {
            α *= settings.core().max_step_fraction;
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

    fn symmetric_initialization(&mut self, cones: &mut CompositeCone<T>) {
        _shift_to_cone_interior(&mut self.s, cones, PrimalOrDualCone::PrimalCone);
        _shift_to_cone_interior(&mut self.z, cones, PrimalOrDualCone::DualCone);

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

    fn barrier(&self, step: &Self, α: T, cones: &mut CompositeCone<T>) -> T {
        let central_coef = (cones.degree() + 1).as_T();

        let cur_τ = self.τ + α * step.τ;
        let cur_κ = self.κ + α * step.κ;

        // compute current μ
        let sz = <[T] as VectorMath>::dot_shifted(&self.z, &self.s, &step.z, &step.s, α);
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

fn _shift_to_cone_interior<T>(z: &mut [T], cones: &mut CompositeCone<T>, pd: PrimalOrDualCone)
where
    T: FloatT,
{
    let (min_margin, pos_margin) = cones.margins(z, pd);
    let target = T::max(
        T::one(),
        (pos_margin * (0.1).as_T()) / cones.degree().as_T(),
    );

    if min_margin <= T::zero() {
        // at least some component is outside its cone
        // done in two stages since otherwise (1-α) = -α for
        // large α, which makes z exactly 0. (or worse, -0.0 )
        cones.scaled_unit_shift(z, -min_margin, pd);
        cones.scaled_unit_shift(z, target, pd);
    } else if min_margin < target {
        // margin is positive but small.
        cones.scaled_unit_shift(z, target - min_margin, pd);
    } else {
        // good margin, but still shift explicitly by
        // zero to catch any elements in the zero cone
        // that need to be forced to zero
        cones.scaled_unit_shift(z, T::zero(), pd);
    }
}
