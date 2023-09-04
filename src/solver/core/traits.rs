//! Required traits for types providing a Clarabel solver implementation.
//!
//! This module defines the core traits that must be implemented by a collection
//! of mutually associated data types to make a solver for a particular problem
//! format.
//!
//! In nearly all cases there is no need for a user to implement these traits.
//! Instead, users should use the collection of types that are provided
//! in the [Default solver implementation](crate::solver::implementations::default),
//!  which collectively implement support for the problem format described in the top
//! level crate documentation.

use super::{cones::Cone, CoreSettings, ScalingStrategy};
use super::{SolverStatus, StepDirection};
use crate::algebra::*;
use crate::timers::*;

/// Data for a conic optimization problem.

pub trait ProblemData<T: FloatT> {
    type V: Variables<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    /// Equilibrate internal data before solver starts.
    fn equilibrate(&mut self, cones: &Self::C, settings: &Self::SE);
}

/// Variables for a conic optimization problem.

pub trait Variables<T: FloatT> {
    type D: ProblemData<T>;
    type R: Residuals<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    /// Compute the scaled duality gap.

    fn calc_mu(&mut self, residuals: &Self::R, cones: &Self::C) -> T;

    /// Compute the KKT RHS for a pure Newton step.

    fn affine_step_rhs(&mut self, residuals: &Self::R, variables: &Self, cones: &Self::C);

    /// Compute the KKT RHS for an interior point centering step.

    #[allow(clippy::too_many_arguments)]
    fn combined_step_rhs(
        &mut self,
        residuals: &Self::R,
        variables: &Self,
        cones: &mut Self::C,
        step: &mut Self, //mut allows step to double as working space
        σ: T,
        μ: T,
        m: T,
    );

    /// Compute the maximum step length possible in the given
    /// step direction without violating a cone boundary.

    fn calc_step_length(
        &self,
        step_lhs: &Self,
        cones: &mut Self::C,
        settings: &Self::SE,
        step_direction: StepDirection,
    ) -> T;

    /// Update the variables in the given step direction, scaled by `α`.
    fn add_step(&mut self, step_lhs: &Self, α: T);

    /// Bring the variables into the interior of the cone constraints.
    fn symmetric_initialization(&mut self, cones: &mut Self::C);

    /// Initialize all conic variables to unit values.
    fn unit_initialization(&mut self, cones: &Self::C);

    /// Overwrite values with those from another object
    fn copy_from(&mut self, src: &Self);

    /// Apply NT scaling to a collection of cones.

    fn scale_cones(&self, cones: &mut Self::C, μ: T, scaling_strategy: ScalingStrategy) -> bool;

    /// Compute the barrier function

    fn barrier(&self, step: &Self, α: T, cones: &mut Self::C) -> T;

    /// Rescale variables, e.g. to renormalize iterates
    /// in a homogeneous embedding

    fn rescale(&mut self);
}

/// Residuals for a conic optimization problem.

pub trait Residuals<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;

    /// Compute residuals for the given variables.
    ///
    fn update(&mut self, variables: &Self::V, data: &Self::D);
}

/// KKT linear solver object.

pub trait KKTSystem<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    /// Update the KKT system.   In particular, update KKT
    /// matrix entries with new variable and refactor.

    fn update(&mut self, data: &Self::D, cones: &Self::C, settings: &Self::SE) -> bool;

    /// Solve the KKT system for the given RHS.

    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        step_lhs: &mut Self::V,
        step_rhs: &Self::V,
        data: &Self::D,
        variables: &Self::V,
        cones: &mut Self::C,
        step_direction: StepDirection,
        settings: &Self::SE,
    ) -> bool;

    /// Find an IP starting condition

    fn solve_initial_point(
        &mut self,
        variables: &mut Self::V,
        data: &Self::D,
        settings: &Self::SE,
    ) -> bool;
}

/// Printing functions for the solver's Info

pub trait InfoPrint<T>
where
    T: FloatT,
{
    type D: ProblemData<T>;
    type C: Cone<T>;
    type SE: Settings<T>;

    /// Print the solver configuration, e.g. settings etc.
    /// This function is called once at the start of the solve.
    fn print_configuration(&self, settings: &Self::SE, data: &Self::D, cones: &Self::C);

    /// Print a header to appear at the top of progress information.
    fn print_status_header(&self, settings: &Self::SE);

    /// Print solver progress information.   Called once per iteration.
    fn print_status(&self, settings: &Self::SE);

    /// Print solver final status and other exit information.   Called at
    /// solver termination.
    fn print_footer(&self, settings: &Self::SE);
}

/// Internal information for the solver to monitor progress and check for termination.

pub trait Info<T>: InfoPrint<T>
where
    T: FloatT,
{
    type V: Variables<T>;
    type R: Residuals<T>;

    /// Reset internal data, particularly solve timers.
    fn reset(&mut self, timers: &mut Timers);

    /// Compute final values before solver termination
    fn finalize(&mut self, residuals: &Self::R, settings: &Self::SE, timers: &mut Timers);

    /// Update solver progress information
    fn update(&mut self, data: &Self::D, variables: &Self::V, residuals: &Self::R, timers: &Timers);

    /// Return `true` if termination conditions have been reached.
    fn check_termination(&mut self, residuals: &Self::R, settings: &Self::SE, iter: u32) -> bool;

    // save and recover prior iterates
    fn save_prev_iterate(&mut self, variables: &Self::V, prev_variables: &mut Self::V);
    fn reset_to_prev_iterate(&mut self, variables: &mut Self::V, prev_variables: &Self::V);

    /// Record some of the top level solver's choice of various
    /// scalars. `μ = ` normalized gap.  `α = ` computed step length.
    /// `σ = ` multiplier for the updated centering parameter.
    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32);

    /// Report or update termination status
    fn get_status(&self) -> SolverStatus;
    fn set_status(&mut self, status: SolverStatus);
}

/// Solution for a conic optimization problem.

pub trait Solution<T: FloatT> {
    type D: ProblemData<T>;
    type V: Variables<T>;
    type I: Info<T>;

    /// Compute solution from the Variables at solver termination
    fn finalize(&mut self, data: &Self::D, variables: &Self::V, info: &Self::I);
}

/// Settings for a conic optimization problem.
///
/// Implementors of this trait can define any internal or problem
/// specific settings they wish.   They must, however, also maintain
/// a settings object of type [`CoreSettings`](crate::solver::core::CoreSettings)
/// and return this to the solver internally.   

pub trait Settings<T: FloatT> {
    /// Return the core settings.
    fn core(&self) -> &CoreSettings<T>;

    /// Return the core settings (mutably).
    fn core_mut(&mut self) -> &mut CoreSettings<T>;
}
