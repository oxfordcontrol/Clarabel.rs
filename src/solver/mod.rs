// internal module structure
pub(crate) mod core;
pub(crate) mod implementations;

//Here we expose only part of the solver internals
//and rearrange public modules a bit to give a more
//user friendly API

//allows declaration of cone constraints
pub use crate::solver::core::cones::{SupportedCones, SupportedCones::*};

//user facing traits required to interact with solver
pub use crate::solver::core::{IPSolver, SolverStatus};

//user facing traits required to define new implementatiions
pub use crate::solver::core::traits;

//If we have implemtations for multple alternative
//problem formats, they would live here.   Since we
//only have default, it is exposed at the top level
//in the use statements directly below instead.

// pub mod implementations {
//     pub mod default {
//         pub use clarabel_solver::implementations::default::*;
//     }
// }

pub use crate::solver::implementations::default::*;

//configure tests of internals
#[cfg(test)]
mod tests;
