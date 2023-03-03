//! Implementation of core types for the standard problem format
//! described in the documentation [main page](crate).

#![allow(non_snake_case)]

mod equilibration;
mod info;
mod info_print;
mod kktsystem;
mod presolver;
mod problemdata;
mod residuals;
mod settings;
mod solution;
mod solver;
mod variables;

// export flattened
pub use equilibration::*;
pub use info::*;
pub use info_print::*;
pub use kktsystem::*;
pub use presolver::*;
pub use problemdata::*;
pub use residuals::*;
pub use settings::*;
pub use solution::*;
pub use solver::*;
pub use variables::*;
