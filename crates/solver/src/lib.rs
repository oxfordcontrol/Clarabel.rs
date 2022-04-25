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
    fn equilibrate(&mut self, cones: &ConeSet<T>, settings: &Settings<T>);
}
