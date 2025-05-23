use crate::solver::core::ffi::*;
use crate::solver::core::traits::Settings;
use crate::{algebra::*, solver::core::SettingsError};
use derive_builder::Builder;

#[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
use pardiso_wrapper::PardisoInterface;
#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[cfg(all(
    feature = "serde",
    any(feature = "pardiso-mkl", feature = "pardiso-panua")
))]
use serde_big_array::BigArray;

// PJG: Serialization is required for file in/out, but is also used to pass
// settings structures between Rust and Julia (and possibly Python)
// Passing to Julia should be done using the new FFI interface types
// implemented in https://github.com/oxfordcontrol/Clarabel.rs/pull/176

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

    ///direct linear solver method(e.g. "faer", "qdldl", "auto")
    #[builder(default = r#""auto".to_string()"#)]
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

    ///explicitly drop structural zeros from sparse data inputs
    ///Caution: this will disable parametric updating functionality
    ///See also ['dropzeros'][crate::algebra::CscMatrix::dropzeros]
    ///for dropping structural zeros before passing to the solver
    ///
    #[builder(default = "false")]
    pub input_sparse_dropzeros: bool,

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

    /// Pardiso `iparm` parameter array.  Any values in this array (other
    /// than iparm[0]) that are set to something other than i32::MIN will be
    /// passed to the solver as is, with all other values taking appropriate
    /// defaults.  
    ///
    /// The parameter iparm[0] is used by pardiso to indicate that default
    /// values should be used everywhere (if 0) or that the user has set
    /// the values in the array (if 1).  Ths parameter will be set internally
    /// to 1 by the solver if the user has passed any values other than i32::MIN.
    ///
    /// Caution: it is the responsibility of the user to ensure that the
    /// values passed are valid for the intended use of Pardiso within the
    /// Clarabel solver.   Failure to do so may result in side effects
    /// ranging from benign to catastrophic.
    ///
    /// NB: depending on the documentation referred to, the 'iparm' array
    /// may be documented using either 0-based (C style) or 1-based (Fortran)
    /// style indexing.  The values in this array are treated as 0-based.  
    ///
    /// Requires the "pardiso-mkl" or "pardiso-panua" feature.
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    #[builder(default = "[i32::MIN; 64]")]
    #[cfg_attr(feature = "serde", serde(with = "BigArray"))]
    pub pardiso_iparm: [i32; 64],

    /// enable pardiso verbose output
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    #[builder(default = "false")]
    pub pardiso_verbose: bool,
}

impl<T> Default for DefaultSettings<T>
where
    T: FloatT,
{
    fn default() -> DefaultSettings<T> {
        DefaultSettingsBuilder::<T>::default().build().unwrap()
    }
}

macro_rules! check_immutable_setting {
    ($self:expr, $prev:expr, $field:ident) => {
        if $self.$field != $prev.$field {
            return Err(SettingsError::ImmutableSetting(stringify!($field)));
        }
    };
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

    /// Checks that the settings are valid.  This only ensures that fields specified
    /// by strings contain valid options.   It does not sanity check numerical values
    fn validate(&self) -> Result<(), SettingsError> {
        // this direct check avoids an internal panic since indirect
        // solvers are not yet available at all
        if !self.direct_kkt_solver {
            return Err(SettingsError::BadFieldValue("direct_kkt_solver"));
        }

        //check that the choice of LDL solver (string) is valid
        validate_direct_solve_method(&self.direct_solve_method)?;

        // check that the chordal decomposition merge method (string) is valid
        #[cfg(feature = "sdp")]
        validate_chordal_decomposition_merge_method(&self.chordal_decomposition_merge_method)?;

        // check the pardiso iparm settings.   Currently a no-op
        #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
        validate_pardiso_iparm(&self.pardiso_iparm)?;

        Ok(())
    }

    /// check that a settings object is valid as an updated collection
    /// of settings for a solver that has already been initialized.   This
    /// should reject changed to parameters that are only applicable during
    /// solver initialization.  Calls `validate()` internally to check
    /// that values are also legal.
    fn validate_as_update(&self, prev: &Self) -> Result<(), SettingsError> {
        self.validate()?;

        check_immutable_setting!(self, prev, equilibrate_enable);
        check_immutable_setting!(self, prev, equilibrate_max_iter);
        check_immutable_setting!(self, prev, equilibrate_min_scaling);
        check_immutable_setting!(self, prev, equilibrate_max_scaling);
        check_immutable_setting!(self, prev, max_threads);
        check_immutable_setting!(self, prev, direct_kkt_solver);
        check_immutable_setting!(self, prev, direct_solve_method);
        check_immutable_setting!(self, prev, presolve_enable);
        check_immutable_setting!(self, prev, input_sparse_dropzeros);

        #[cfg(feature = "sdp")]
        {
            check_immutable_setting!(self, prev, chordal_decomposition_enable);
            check_immutable_setting!(self, prev, chordal_decomposition_merge_method);
            check_immutable_setting!(self, prev, chordal_decomposition_compact);
            check_immutable_setting!(self, prev, chordal_decomposition_complete_dual);
        }

        #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
        {
            check_immutable_setting!(self, prev, pardiso_iparm);
            check_immutable_setting!(self, prev, pardiso_verbose);
        }

        Ok(())
    }
}

impl<T: FloatT> ClarabelFFI<Self> for DefaultSettings<T> {
    type FFI = super::ffi::DefaultSettingsFFI<T>;
}

// pre build checker (for auto-validation when using the builder)

impl From<SettingsError> for DefaultSettingsBuilderError {
    fn from(e: SettingsError) -> Self {
        DefaultSettingsBuilderError::ValidationError(e.to_string())
    }
}

/// Automatic pre-build settings validation
impl<T> DefaultSettingsBuilder<T>
where
    T: FloatT,
{
    /// check that the specified direct_solve_method is valid
    pub fn validate(&self) -> Result<(), SettingsError> {
        if let Some(ref direct_solve_method) = self.direct_solve_method {
            validate_direct_solve_method(direct_solve_method)?;
        }

        // check that the chordal decomposition merge method is valid
        #[cfg(feature = "sdp")]
        if let Some(ref chordal_decomposition_merge_method) =
            self.chordal_decomposition_merge_method
        {
            validate_chordal_decomposition_merge_method(chordal_decomposition_merge_method)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------
// individual validation functions go here
// ---------------------------------------------------------

fn validate_direct_solve_method(direct_solve_method: &str) -> Result<(), SettingsError> {
    match direct_solve_method {
        "auto" => Ok(()),
        "qdldl" => Ok(()),
        #[cfg(feature = "faer-sparse")]
        "faer" => Ok(()),
        #[cfg(feature = "pardiso-mkl")]
        "mkl" => {
            if pardiso_wrapper::MKLPardisoSolver::is_available() {
                Ok(())
            } else {
                Err(SettingsError::LinearSolverProblem {
                    solver: "mkl",
                    problem: "not available",
                })
            }
        }
        #[cfg(feature = "pardiso-panua")]
        "panua" => {
            if pardiso_wrapper::PanuaPardisoSolver::is_available() {
                Ok(())
            } else {
                Err(SettingsError::LinearSolverProblem {
                    solver: "panua",
                    problem: "not available",
                })
            }
        }
        _ => Err(SettingsError::BadFieldValue("direct_solve_method")),
    }
}

#[cfg(feature = "sdp")]
fn validate_chordal_decomposition_merge_method(
    chordal_decomposition_merge_method: &str,
) -> Result<(), SettingsError> {
    match chordal_decomposition_merge_method {
        "none" => Ok(()),
        "parent_child" => Ok(()),
        "clique_graph" => Ok(()),
        _ => Err(SettingsError::BadFieldValue(
            "chordal_decomposition_merge_method",
        )),
    }
}

#[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
fn validate_pardiso_iparm(iparm: &[i32; 64]) -> Result<(), SettingsError> {
    use crate::solver::core::kktsolvers::direct::ldlsolvers::pardiso::pardiso_iparm_is_valid;

    // check for very bad parm parameters here

    if pardiso_iparm_is_valid(iparm) {
        Ok(())
    } else {
        Err(SettingsError::BadFieldValue("pardiso_iparm"))
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

    // directly construct a bad DefaultSettings and manually check
    let settings = DefaultSettings::<f64> {
        direct_solve_method: "foo".to_string(),
        ..DefaultSettings::default()
    };
    assert!(settings.validate().is_err());

    // try to overlay prohibited update values
    let oldsettings = DefaultSettings::<f64> {
        presolve_enable: false,
        ..DefaultSettings::default()
    };

    let newsettings = DefaultSettings::<f64> {
        presolve_enable: true,
        ..DefaultSettings::default()
    };
    assert!(newsettings.validate_as_update(&oldsettings).is_err());

    // try to overlay allowed update values
    let oldsettings = DefaultSettings::<f64> {
        max_iter: 10,
        ..DefaultSettings::default()
    };

    let newsettings = DefaultSettings::<f64> {
        max_iter: 11,
        ..DefaultSettings::default()
    };
    assert!(newsettings.validate_as_update(&oldsettings).is_ok());
}
