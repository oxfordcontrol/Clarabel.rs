use crate::solver::implementations::default::DefaultSettings;
use thiserror::Error;

/// Solver general core settings are the same as in the default solver.
///
/// Go [here](crate::solver::implementations::default::DefaultSettings)
/// to view the complete list.
///
pub type CoreSettings<T> = DefaultSettings<T>;

#[derive(Error, Debug)]
/// Error type returned by settings validation
pub enum SettingsError {
    /// An error attributable to one of the fields
    #[error("Bad field")]
    BadField(&'static str),
    /// a subsolver error of some kind (e.g. not found, no license)
    #[error("Problem with {solver} solver ({problem})")]
    LinearSolverProblem {
        solver: &'static str,
        problem: &'static str,
    },
}
