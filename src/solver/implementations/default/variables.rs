use super::*;
use crate::algebra::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone, PrimalOrDualCone},
    traits::{Settings, Variables},
    ScalingStrategy, StepDirection,
};
use thiserror::Error;

// ---------------
// Error type for variable initialization
// ---------------

/// Error type returned by variable initialization operations.
#[derive(Error, Debug)]
pub enum VariableInitializationError {
    /// Slack variables must be positive (s > 0)
    #[error("Slack variables must be positive (s > 0)")]
    InvalidSlackVariables,
    /// Dual variables must be positive (z > 0)
    #[error("Dual variables must be positive (z > 0)")]
    InvalidDualVariables,
    /// Homogenization scalar τ must be positive (τ > 0)
    #[error("Homogenization scalar τ must be positive (τ > 0)")]
    InvalidTau,
    /// Homogenization scalar κ must be positive (κ > 0)
    #[error("Homogenization scalar κ must be positive (κ > 0)")]
    InvalidKappa,
    /// Dimension mismatch for x vector
    #[error("Dimension mismatch: expected x length {expected}, got {actual}")]
    XDimensionMismatch { 
        /// Expected length
        expected: usize, 
        /// Actual length
        actual: usize 
    },
    /// Dimension mismatch for s vector
    #[error("Dimension mismatch: expected s length {expected}, got {actual}")]
    SDimensionMismatch { 
        /// Expected length
        expected: usize, 
        /// Actual length
        actual: usize 
    },
    /// Dimension mismatch for z vector
    #[error("Dimension mismatch: expected z length {expected}, got {actual}")]
    ZDimensionMismatch { 
        /// Expected length
        expected: usize, 
        /// Actual length
        actual: usize 
    },
}

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
    /// Create a new `DefaultVariables` object
    pub fn new(n: usize, m: usize) -> Self {
        let x = vec![T::zero(); n];
        let s = vec![T::zero(); m];
        let z = vec![T::zero(); m];
        let τ = T::one();
        let κ = T::one();

        Self { x, s, z, τ, κ }
    }

    /// Initialize variables with user-provided values and validation
    /// 
    /// This function allows partial initialization of variables while performing
    /// basic validity checks:
    /// - All components of `s` must be positive (s > 0)
    /// - All components of `z` must be positive (z > 0)  
    /// - `τ` must be positive (τ > 0)
    /// - `κ` must be positive (κ > 0)
    /// 
    /// If a parameter is `None`, the corresponding variable is initialized with default values:
    /// - `x` defaults to zeros
    /// - `s` defaults to ones
    /// - `z` defaults to ones
    /// - `τ` defaults to 1.0
    /// - `κ` defaults to 1.0
    /// 
    /// # Arguments
    /// 
    /// * `x` - Optional primal variables (can be any value)
    /// * `s` - Optional slack variables (must be positive if provided)
    /// * `z` - Optional dual variables (must be positive if provided)
    /// * `τ` - Optional homogenization scalar τ (must be positive if provided)
    /// * `κ` - Optional homogenization scalar κ (must be positive if provided)
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - Initialization successful
    /// * `Err(VariableInitializationError)` - Validation failed or dimension mismatch
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use clarabel::solver::DefaultVariables;
    /// 
    /// // Create variables with dimensions n=3, m=2
    /// let mut vars = DefaultVariables::<f64>::new(3, 2);
    /// 
    /// // Full initialization with validation
    /// let x_vals = vec![1.0, 2.0, 3.0];
    /// let s_vals = vec![0.5, 1.5]; // must be positive
    /// let z_vals = vec![2.0, 3.0]; // must be positive
    /// 
    /// vars.initialize_with_values(
    ///     Some(&x_vals),
    ///     Some(&s_vals),
    ///     Some(&z_vals),
    ///     Some(2.0), // τ must be positive
    ///     Some(1.5), // κ must be positive
    /// ).expect("Initialization should succeed");
    /// 
    /// // Partial initialization (only x and τ)
    /// vars.initialize_with_values(
    ///     Some(&[10.0, 20.0, 30.0]),
    ///     None, // s defaults to ones
    ///     None, // z defaults to ones
    ///     Some(0.8),
    ///     None, // κ defaults to one
    /// ).expect("Partial initialization should succeed");
    /// ```
    /// 
    /// # Errors
    /// 
    /// This function will return an error if:
    /// - Any component of `s` is ≤ 0
    /// - Any component of `z` is ≤ 0  
    /// - `τ` is ≤ 0
    /// - `κ` is ≤ 0
    /// - Vector dimensions don't match the variable dimensions
    pub fn initialize_with_values(
        &mut self,
        x: Option<&[T]>,
        s: Option<&[T]>,
        z: Option<&[T]>,
        τ: Option<T>,
        κ: Option<T>,
    ) -> Result<(), VariableInitializationError> {
        // Validate and set x if provided
        if let Some(x_vals) = x {
            if x_vals.len() != self.x.len() {
                return Err(VariableInitializationError::XDimensionMismatch {
                    expected: self.x.len(),
                    actual: x_vals.len(),
                });
            }
            self.x.copy_from_slice(x_vals);
        } else {
            // Default to zeros
            self.x.fill(T::zero());
        }

        // Validate and set s if provided
        if let Some(s_vals) = s {
            if s_vals.len() != self.s.len() {
                return Err(VariableInitializationError::SDimensionMismatch {
                    expected: self.s.len(),
                    actual: s_vals.len(),
                });
            }
            // Check all components are positive
            for &val in s_vals {
                if val <= T::zero() {
                    return Err(VariableInitializationError::InvalidSlackVariables);
                }
            }
            self.s.copy_from_slice(s_vals);
        } else {
            // Default to ones
            self.s.fill(T::one());
        }

        // Validate and set z if provided
        if let Some(z_vals) = z {
            if z_vals.len() != self.z.len() {
                return Err(VariableInitializationError::ZDimensionMismatch {
                    expected: self.z.len(),
                    actual: z_vals.len(),
                });
            }
            // Check all components are positive
            for &val in z_vals {
                if val <= T::zero() {
                    return Err(VariableInitializationError::InvalidDualVariables);
                }
            }
            self.z.copy_from_slice(z_vals);
        } else {
            // Default to ones
            self.z.fill(T::one());
        }

        // Validate and set τ if provided
        if let Some(tau_val) = τ {
            if tau_val <= T::zero() {
                return Err(VariableInitializationError::InvalidTau);
            }
            self.τ = tau_val;
        } else {
            // Default to one
            self.τ = T::one();
        }

        // Validate and set κ if provided
        if let Some(kappa_val) = κ {
            if kappa_val <= T::zero() {
                return Err(VariableInitializationError::InvalidKappa);
            }
            self.κ = kappa_val;
        } else {
            // Default to one
            self.κ = T::one();
        }

        Ok(())
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

impl<T> DefaultVariables<T>
where
    T: FloatT,
{
    pub(crate) fn unscale(&mut self, data: &DefaultProblemData<T>, is_infeasible: bool) {
        // if we have an infeasible problem, normalize
        // using κ to get an infeasibility certificate.
        // Otherwise use τ to get an unscaled solution.
        let scaleinv = {
            if is_infeasible {
                T::recip(self.κ)
            } else {
                T::recip(self.τ)
            }
        };

        // also undo the equilibration
        let d = &data.equilibration.d;
        let (e, einv) = (&data.equilibration.e, &data.equilibration.einv);
        let cinv = T::recip(data.equilibration.c);

        self.x.hadamard(d).scale(scaleinv);
        self.z.hadamard(e).scale(scaleinv * cinv);
        self.s.hadamard(einv).scale(scaleinv);

        self.τ *= scaleinv;
        self.κ *= scaleinv;
    }

    #[cfg_attr(not(feature = "sdp"), allow(dead_code))]
    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.x.len(), self.s.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialize_with_values_success() {
        let mut vars = DefaultVariables::<f64>::new(3, 2);

        // Test successful initialization with all parameters
        let x_vals = vec![1.0, 2.0, 3.0];
        let s_vals = vec![0.5, 1.5];
        let z_vals = vec![2.0, 3.0];
        let tau_val = 2.0;
        let kappa_val = 1.5;

        let result = vars.initialize_with_values(
            Some(&x_vals),
            Some(&s_vals),
            Some(&z_vals),
            Some(tau_val),
            Some(kappa_val),
        );

        assert!(result.is_ok());
        assert_eq!(vars.x, x_vals);
        assert_eq!(vars.s, s_vals);
        assert_eq!(vars.z, z_vals);
        assert_eq!(vars.τ, tau_val);
        assert_eq!(vars.κ, kappa_val);
    }

    #[test]
    fn test_initialize_with_values_partial() {
        let mut vars = DefaultVariables::<f64>::new(2, 2);

        // Test partial initialization (only x and τ)
        let x_vals = vec![1.0, 2.0];
        let tau_val = 0.5;

        let result = vars.initialize_with_values(
            Some(&x_vals),
            None,        // s defaults to ones
            None,        // z defaults to ones
            Some(tau_val),
            None,        // κ defaults to one
        );

        assert!(result.is_ok());
        assert_eq!(vars.x, x_vals);
        assert_eq!(vars.s, vec![1.0, 1.0]); // default
        assert_eq!(vars.z, vec![1.0, 1.0]); // default
        assert_eq!(vars.τ, tau_val);
        assert_eq!(vars.κ, 1.0); // default
    }

    #[test]
    fn test_initialize_with_values_invalid_s() {
        let mut vars = DefaultVariables::<f64>::new(2, 2);

        // Test with negative slack variable
        let s_vals = vec![1.0, -0.5]; // negative value should fail

        let result = vars.initialize_with_values(
            None,
            Some(&s_vals),
            None,
            None,
            None,
        );

        assert!(matches!(
            result,
            Err(VariableInitializationError::InvalidSlackVariables)
        ));
    }

    #[test]
    fn test_initialize_with_values_invalid_z() {
        let mut vars = DefaultVariables::<f64>::new(2, 2);

        // Test with zero dual variable
        let z_vals = vec![1.0, 0.0]; // zero value should fail

        let result = vars.initialize_with_values(
            None,
            None,
            Some(&z_vals),
            None,
            None,
        );

        assert!(matches!(
            result,
            Err(VariableInitializationError::InvalidDualVariables)
        ));
    }

    #[test]
    fn test_initialize_with_values_invalid_tau() {
        let mut vars = DefaultVariables::<f64>::new(2, 2);

        // Test with negative τ
        let result = vars.initialize_with_values(
            None,
            None,
            None,
            Some(-1.0),
            None,
        );

        assert!(matches!(
            result,
            Err(VariableInitializationError::InvalidTau)
        ));
    }

    #[test]
    fn test_initialize_with_values_invalid_kappa() {
        let mut vars = DefaultVariables::<f64>::new(2, 2);

        // Test with zero κ
        let result = vars.initialize_with_values(
            None,
            None,
            None,
            None,
            Some(0.0),
        );

        assert!(matches!(
            result,
            Err(VariableInitializationError::InvalidKappa)
        ));
    }

    #[test]
    fn test_initialize_with_values_dimension_mismatch() {
        let mut vars = DefaultVariables::<f64>::new(3, 2);

        // Test with wrong x dimension
        let x_vals = vec![1.0, 2.0]; // should be length 3

        let result = vars.initialize_with_values(
            Some(&x_vals),
            None,
            None,
            None,
            None,
        );

        assert!(matches!(
            result,
            Err(VariableInitializationError::XDimensionMismatch { expected: 3, actual: 2 })
        ));

        // Test with wrong s dimension
        let s_vals = vec![1.0, 2.0, 3.0]; // should be length 2

        let result = vars.initialize_with_values(
            None,
            Some(&s_vals),
            None,
            None,
            None,
        );

        assert!(matches!(
            result,
            Err(VariableInitializationError::SDimensionMismatch { expected: 2, actual: 3 })
        ));
    }
}
