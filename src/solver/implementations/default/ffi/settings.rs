use crate::algebra::*;
use crate::solver::core::ffi::*;
use crate::solver::DefaultSettings;

/// FFI interface for [`DefaultSettings`](crate::solver::implementations::default::DefaultSettings)
#[allow(missing_docs)]
#[derive(Debug, Clone)]
#[repr(C)]
pub struct DefaultSettingsFFI<T: FloatT> {
    // Main algorithm settings
    pub max_iter: u32,
    pub time_limit: f64,
    pub verbose: bool,
    pub max_step_fraction: T,

    // Full accuracy settings
    pub tol_gap_abs: T,
    pub tol_gap_rel: T,
    pub tol_feas: T,
    pub tol_infeas_abs: T,
    pub tol_infeas_rel: T,
    pub tol_ktratio: T,

    // Reduced accuracy settings
    pub reduced_tol_gap_abs: T,
    pub reduced_tol_gap_rel: T,
    pub reduced_tol_feas: T,
    pub reduced_tol_infeas_abs: T,
    pub reduced_tol_infeas_rel: T,
    pub reduced_tol_ktratio: T,

    // data equilibration settings
    pub equilibrate_enable: bool,
    pub equilibrate_max_iter: u32,
    pub equilibrate_min_scaling: T,
    pub equilibrate_max_scaling: T,

    // Step size settings
    pub linesearch_backtrack_step: T,
    pub min_switch_step_length: T,
    pub min_terminate_step_length: T,

    // Linear solver settings
    pub max_threads: u32,
    pub direct_kkt_solver: bool,
    pub direct_solve_method: DirectSolveMethodsFFI,

    // static regularization parameters
    pub static_regularization_enable: bool,
    pub static_regularization_constant: T,
    pub static_regularization_proportional: T,

    // dynamic regularization parameters
    pub dynamic_regularization_enable: bool,
    pub dynamic_regularization_eps: T,
    pub dynamic_regularization_delta: T,

    // iterative refinement (for direct solves)
    pub iterative_refinement_enable: bool,
    pub iterative_refinement_reltol: T,
    pub iterative_refinement_abstol: T,
    pub iterative_refinement_max_iter: u32,
    pub iterative_refinement_stop_ratio: T,

    // preprocessing
    pub presolve_enable: bool,
    pub input_sparse_dropzeros: bool,

    // chordal decomposition
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_enable: bool,
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_merge_method: CliqueMergeMethodsFFI,
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_compact: bool,
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_complete_dual: bool,

    //pardiso settings
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    pub pardiso_iparm: [i32; 64],
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    pub pardiso_verbose: bool,
}

// implement From in both directions, since we need to both send
// and receive settings from the solver FFI interface.
// Some kind of procedural macro would be nice here.  derive_more
// seemingly doesn't want to apply "into" on the subfields.

macro_rules! impl_from {
    ($A:ident, $B:ident) => {
        impl<T> From<$A<T>> for $B<T>
        where
            T: FloatT,
        {
            fn from(settings: $A<T>) -> Self {
                Self {
                    max_iter: settings.max_iter,
                    time_limit: settings.time_limit,
                    verbose: settings.verbose,
                    max_step_fraction: settings.max_step_fraction,
                    tol_gap_abs: settings.tol_gap_abs,
                    tol_gap_rel: settings.tol_gap_rel,
                    tol_feas: settings.tol_feas,
                    tol_infeas_abs: settings.tol_infeas_abs,
                    tol_infeas_rel: settings.tol_infeas_rel,
                    tol_ktratio: settings.tol_ktratio,
                    reduced_tol_gap_abs: settings.reduced_tol_gap_abs,
                    reduced_tol_gap_rel: settings.reduced_tol_gap_rel,
                    reduced_tol_feas: settings.reduced_tol_feas,
                    reduced_tol_infeas_abs: settings.reduced_tol_infeas_abs,
                    reduced_tol_infeas_rel: settings.reduced_tol_infeas_rel,
                    reduced_tol_ktratio: settings.reduced_tol_ktratio,
                    equilibrate_enable: settings.equilibrate_enable,
                    equilibrate_max_iter: settings.equilibrate_max_iter,
                    equilibrate_min_scaling: settings.equilibrate_min_scaling,
                    equilibrate_max_scaling: settings.equilibrate_max_scaling,
                    linesearch_backtrack_step: settings.linesearch_backtrack_step,
                    min_switch_step_length: settings.min_switch_step_length,
                    min_terminate_step_length: settings.min_terminate_step_length,
                    max_threads: settings.max_threads,
                    direct_kkt_solver: settings.direct_kkt_solver,
                    direct_solve_method: settings.direct_solve_method.into(),
                    static_regularization_enable: settings.static_regularization_enable,
                    static_regularization_constant: settings.static_regularization_constant,
                    static_regularization_proportional: settings.static_regularization_proportional,
                    dynamic_regularization_enable: settings.dynamic_regularization_enable,
                    dynamic_regularization_eps: settings.dynamic_regularization_eps,
                    dynamic_regularization_delta: settings.dynamic_regularization_delta,
                    iterative_refinement_enable: settings.iterative_refinement_enable,
                    iterative_refinement_reltol: settings.iterative_refinement_reltol,
                    iterative_refinement_abstol: settings.iterative_refinement_abstol,
                    iterative_refinement_max_iter: settings.iterative_refinement_max_iter,
                    iterative_refinement_stop_ratio: settings.iterative_refinement_stop_ratio,
                    presolve_enable: settings.presolve_enable,
                    input_sparse_dropzeros: settings.input_sparse_dropzeros,
                    #[cfg(feature = "sdp")]
                    chordal_decomposition_enable: settings.chordal_decomposition_enable,
                    #[cfg(feature = "sdp")]
                    chordal_decomposition_merge_method: settings
                        .chordal_decomposition_merge_method
                        .into(),
                    #[cfg(feature = "sdp")]
                    chordal_decomposition_compact: settings.chordal_decomposition_compact,
                    #[cfg(feature = "sdp")]
                    chordal_decomposition_complete_dual: settings
                        .chordal_decomposition_complete_dual,
                    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
                    pardiso_iparm: settings.pardiso_iparm,
                    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
                    pardiso_verbose: settings.pardiso_verbose,
                }
            }
        }
    };
}

// implement From in both directions
// DefaultSettingsFFI -> DefaultSettings
impl_from!(DefaultSettingsFFI, DefaultSettings);
impl_from!(DefaultSettings, DefaultSettingsFFI);

#[test]
fn test_settings_ffi() {
    use super::*;

    let settings = DefaultSettings::<f64> {
        max_iter: 123,
        ..Default::default()
    };
    let settings_ffi: DefaultSettingsFFI<f64> = settings.clone().into();

    assert_eq!(settings.max_iter, settings_ffi.max_iter);
}
