//PJG: Temporary allows during development
#![allow(confusable_idents)]
#![allow(dead_code)]
// #![allow(unused_variables)]
// #![allow(unused_mut)]

//PJG: Are both use and mod required throughout?
//Need a general cleanup of these header like
//declarations

pub use clarabel_algebra as algebra;

pub mod cones;
pub mod conicvector;
pub mod default;
pub mod kktsolvers;
pub mod settings;
pub mod solver;
pub use cones::*;
pub use settings::Settings;

use algebra::*;

pub enum SolverStatus {
    Unsolved,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterations,
    MaxTime,
}

pub trait ProblemData<T: FloatT> {
    //associated types
    type V; //should impl Variables

    fn equilibrate(&mut self, cones: &ConeSet<T>, settings: &Settings<T>);
}

pub trait Variables<T: FloatT> {
    //associated types
    type D: ProblemData<T>;
    type R: Residuals<T>;

    fn calc_mu(&mut self, residuals: &Self::R, cones: &ConeSet<T>) -> T;
    fn calc_affine_step_rhs(
        &mut self,
        residuals: &Self::R,
        data: &Self::D,
        variables: &Self,
        cones: &ConeSet<T>,
    );
    fn calc_combined_step_rhs(
        &mut self,
        residuals: &Self::R,
        data: &Self::D,
        variables: &Self,
        cones: &ConeSet<T>,
        step_lhs: &Self,
        σ: T,
        μ: T,
    );
    fn calc_step_length(&mut self, step_lhs: &Self, cones: &ConeSet<T>) -> T;
    fn add_step(&mut self, step_lhs: &Self, α: T);
    fn shift_to_cone(&mut self, cones: &ConeSet<T>);
}

pub trait Residuals<T: FloatT> {
    //associated types
    type D: ProblemData<T>;
    type V: Variables<T>;

    fn update(&mut self, variables: &Self::V, data: &Self::D);
}

pub trait KKTSystem<T: FloatT> {
    //associated types
    type D: ProblemData<T>; //should impl ProblemData
    type V: Variables<T>; //should impl Variables

    fn update(&mut self, data: &Self::D, cones: &ConeSet<T>);

    //PJG: stepstype as string sucks here
    fn solve(
        &mut self,
        step_lhs: &Self::V,
        step_rhs: &Self::V,
        data: &Self::D,
        variables: &Self::V,
        cones: &ConeSet<T>,
        steptype: &str,
    );

    fn solve_initial_point(&mut self, variables: &Self::V, data: &Self::D);
}

pub trait SolveInfo<T: FloatT> {
    //associated types
    type D: ProblemData<T>;
    type V: Variables<T>;
    type R: Residuals<T>;

    fn reset(&mut self);
    fn finalize(&mut self);

    fn print_header(&mut self, settings: &Settings<T>, data: &Self::D, cones: &ConeSet<T>);
    fn print_status(&mut self, settings: &Settings<T>);
    fn print_footer(&mut self, settings: &Settings<T>);

    fn update(
        &mut self,
        data: &Self::D,
        variables: &Self::V,
        residuals: &Self::R,
        settings: &Settings<T>,
    );

    fn check_termination(&mut self, residuals: &Self::R, settings: &Settings<T>) -> bool;

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: i32);
}

pub trait SolveResult<T: FloatT> {
    //associated types
    type D: ProblemData<T>;
    type V: Variables<T>;
    type SI: SolveInfo<T>;

    fn finalize(&mut self, data: &Self::D, variables: &Self::V, info: &Self::SI);
}
