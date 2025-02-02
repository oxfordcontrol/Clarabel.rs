use crate::algebra::*;
use crate::solver::core::traits::Settings;
use derive_builder::Builder;

#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Standard-form solver type implementing the [`Settings`](crate::solver::core::traits::Settings) trait

#[derive(Builder, Debug, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: Serialize + DeserializeOwned"))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct DefaultSettings<T: FloatT> {
    ///maximum number of iterations
    #[builder(default = "200")]
    pub max_iter: u32,

    ///maximum run time (seconds)
    #[builder(default = "f64::INFINITY")]
    pub time_limit: f64,

    ///verbose printing
    #[builder(default = "true")]
    pub verbose: bool,

    ///maximum interior point step length
    #[builder(default = "(0.99).as_T()")]
    pub max_step_fraction: T,

    ///absolute duality gap tolerance
    #[builder(default = "(1e-8).as_T()")]
    pub tol_gap_abs: T,

    ///relative duality gap tolerance
    #[builder(default = "(1e-8).as_T()")]
    pub tol_gap_rel: T,

    ///feasibility check tolerance (primal and dual)
    #[builder(default = "(1e-8).as_T()")]
    pub tol_feas: T,

    ///absolute infeasibility tolerance (primal and dual)
    #[builder(default = "(1e-8).as_T()")]
    pub tol_infeas_abs: T,

    ///relative infeasibility tolerance (primal and dual)
    #[builder(default = "(1e-8).as_T()")]
    pub tol_infeas_rel: T,

    ///κ/τ tolerance
    #[builder(default = "(1e-6).as_T()")]
    pub tol_ktratio: T,

    ///reduced absolute duality gap tolerance
    // NB: reduced_tol_infeas_abs is *smaller* when relaxed, since
    // we are checking that we are this far into the interior of
    // an inequality when checking.   Smaller for this value means
    // "less margin required"
    #[builder(default = "(5e-5).as_T()")]
    pub reduced_tol_gap_abs: T,

    ///reduced relative duality gap tolerance
    #[builder(default = "(5e-5).as_T()")]
    pub reduced_tol_gap_rel: T,

    ///reduced feasibility check tolerance (primal and dual)
    #[builder(default = "(1e-4).as_T()")]
    pub reduced_tol_feas: T,

    ///reduced absolute infeasibility tolerance (primal and dual)
    #[builder(default = "(5e-12).as_T()")]
    pub reduced_tol_infeas_abs: T,

    ///reduced relative infeasibility tolerance (primal and dual)
    #[builder(default = "(5e-5).as_T()")]
    pub reduced_tol_infeas_rel: T,

    ///reduced κ/τ tolerance
    #[builder(default = "(1e-4).as_T()")]
    pub reduced_tol_ktratio: T,

    ///enable data equilibration pre-scaling
    #[builder(default = "true")]
    pub equilibrate_enable: bool,

    /// maximum equilibration scaling iterations
    #[builder(default = "10")]
    pub equilibrate_max_iter: u32,

    ///minimum equilibration scaling allowed
    #[builder(default = "(1e-4).as_T()")]
    pub equilibrate_min_scaling: T,

    ///maximum equilibration scaling allowed
    #[builder(default = "(1e+4).as_T()")]
    pub equilibrate_max_scaling: T,

    ///line search backtracking
    #[builder(default = "(0.8).as_T()")]
    pub linesearch_backtrack_step: T,

    ///minimum step size allowed for asymmetric cones with PrimalDual scaling
    #[builder(default = "(1e-1).as_T()")]
    pub min_switch_step_length: T,

    ///minimum step size allowed for symmetric cones & asymmetric cones with Dual scaling
    #[builder(default = "(1e-4).as_T()")]
    pub min_terminate_step_length: T,

    ///maximum solver threads for multithreaded KKT solvers
    ///choosing 0 lets the solver choose for itself
    #[builder(default = "0")]
    pub max_threads: u32,

    ///use a direct linear solver method (required true)
    #[builder(default = "true")]
    pub direct_kkt_solver: bool,

    ///direct linear solver (e.g. "qdldl")
    #[builder(default = r#""qdldl".to_string()"#)]
    pub direct_solve_method: String,

    ///enable KKT static regularization
    #[builder(default = "true")]
    pub static_regularization_enable: bool,

    ///KKT static regularization parameter
    #[builder(default = "(1e-8).as_T()")]
    pub static_regularization_constant: T,

    ///additional regularization parameter w.r.t. the maximum abs diagonal term
    #[builder(default = "T::epsilon()*T::epsilon()")]
    pub static_regularization_proportional: T,

    ///enable KKT dynamic regularization
    #[builder(default = "true")]
    pub dynamic_regularization_enable: bool,

    ///KKT dynamic regularization threshold
    #[builder(default = "(1e-13).as_T()")]
    pub dynamic_regularization_eps: T,

    ///KKT dynamic regularization shift
    #[builder(default = "(2e-7).as_T()")]
    pub dynamic_regularization_delta: T,

    ///KKT direct solve with iterative refinement
    #[builder(default = "true")]
    pub iterative_refinement_enable: bool,

    ///iterative refinement relative tolerance
    #[builder(default = "(1e-13).as_T()")]
    pub iterative_refinement_reltol: T,

    ///iterative refinement absolute tolerance
    #[builder(default = "(1e-12).as_T()")]
    pub iterative_refinement_abstol: T,

    ///iterative refinement maximum iterations
    #[builder(default = "10")]
    pub iterative_refinement_max_iter: u32,

    ///iterative refinement stalling tolerance
    #[builder(default = "(5.0).as_T()")]
    pub iterative_refinement_stop_ratio: T,

    ///enable presolve constraint reduction
    #[builder(default = "true")]
    pub presolve_enable: bool,

    /// enable chordal decomposition.
    /// [requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = "true")]
    pub chordal_decomposition_enable: bool,

    ///chordal decomposition merge method ("none", "parent_child" or "clique_graph").  
    /// [requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = r#""clique_graph".to_string()"#)]
    pub chordal_decomposition_merge_method: String,

    ///assemble decomposed system in "compact" form
    ///[requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = "true")]
    pub chordal_decomposition_compact: bool,

    ///complete PSD dual variables after decomposition
    /// [requires "sdp" feature.]
    #[cfg(feature = "sdp")]
    #[builder(default = "true")]
    pub chordal_decomposition_complete_dual: bool,
}

impl<T> Default for DefaultSettings<T>
where
    T: FloatT,
{
    fn default() -> DefaultSettings<T> {
        DefaultSettingsBuilder::<T>::default().build().unwrap()
    }
}

impl<T> Settings<T> for DefaultSettings<T>
where
    T: FloatT,
{
    //NB: CoreSettings is typedef'd to DefaultSettings
    fn core(&self) -> &DefaultSettings<T> {
        self
    }
    fn core_mut(&mut self) -> &mut DefaultSettings<T> {
        self
    }
}

// pre build checker (for auto-validation when using the builder)

/// Automatic pre-build settings validation
impl<T> DefaultSettingsBuilder<T>
where
    T: FloatT,
{
    /// check that the specified direct_solve_method is valid
    pub fn validate(&self) -> Result<(), String> {
        if let Some(ref direct_solve_method) = self.direct_solve_method {
            validate_direct_solve_method(direct_solve_method.as_str())?;
        }

        // check that the chordal decomposition merge method is valid
        #[cfg(feature = "sdp")]
        if let Some(ref chordal_decomposition_merge_method) =
            self.chordal_decomposition_merge_method
        {
            validate_chordal_decomposition_merge_method(
                chordal_decomposition_merge_method.as_str(),
            )?;
        }

        Ok(())
    }
}

// post build checker (for ad-hoc validation, e.g. when passing from python/Julia)
// this is not used directly in the solver, but can be called manually by the user

/// Manual post-build settings validation
impl<T> DefaultSettings<T>
where
    T: FloatT,
{
    /// Checks that the settings are valid
    pub fn validate(&self) -> Result<(), String> {
        validate_direct_solve_method(&self.direct_solve_method)?;

        // check that the chordal decomposition merge method is valid
        #[cfg(feature = "sdp")]
        validate_chordal_decomposition_merge_method(&self.chordal_decomposition_merge_method)?;

        Ok(())
    }
}

// ---------------------------------------------------------
// individual validation functions go here
// ---------------------------------------------------------

fn validate_direct_solve_method(direct_solve_method: &str) -> Result<(), String> {
    match direct_solve_method {
        "qdldl" => Ok(()),
        #[cfg(feature = "faer-sparse")]
        "faer" => Ok(()),
        _ => Err(format!(
            "Invalid direct_solve_method: {:?}",
            direct_solve_method
        )),
    }
}

#[cfg(feature = "sdp")]
fn validate_chordal_decomposition_merge_method(
    chordal_decomposition_merge_method: &str,
) -> Result<(), String> {
    match chordal_decomposition_merge_method {
        "none" => Ok(()),
        "parent_child" => Ok(()),
        "clique_graph" => Ok(()),
        _ => Err(format!(
            "Invalid chordal_decomposition_merge_method: {}",
            chordal_decomposition_merge_method
        )),
    }
}

#[test]
fn test_settings_validate() {
    // all standard settings
    DefaultSettingsBuilder::<f64>::default().build().unwrap();

    // fail on unknown direct solve method
    assert!(DefaultSettingsBuilder::<f64>::default()
        .direct_solve_method("foo".to_string())
        .build()
        .is_err());

    // fail on solve options in disabled feature
    let builder = DefaultSettingsBuilder::<f64>::default()
        .direct_solve_method("faer".to_string())
        .build();
    cfg_if::cfg_if! {
        if #[cfg(feature = "faer-sparse")] {
            assert!(builder.is_ok());
        }
        else {
            assert!(builder.is_err());
        }
    }

    #[cfg(feature = "sdp")]
    // fail on unknown chordal decomposition merge method
    assert!(DefaultSettingsBuilder::<f64>::default()
        .chordal_decomposition_merge_method("foo".to_string())
        .build()
        .is_err());
}
