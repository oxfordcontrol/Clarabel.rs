//! Clarabel solver main module.
//!
//! This module contains the main types and traits for the Clarabel solver.
//!
//! The solver comes with a [default implementation](crate::solver::implementations::default)
//! of all required traits.   This produces a solver that solves problems in the standard format
//! described in the top level [API page](crate) and the
//! [User's guide](https://oxfordcontrol.github.io/ClarabelDocs/stable).   This implementation
//! is the most appropriate choice for nearly all users.
//!
//! It is also possible to implement a custom solver by defining a collection
//! of custom types that together implement all of the required core
//! [traits] for objects in Clarabel's core solver.
// internal module structure
pub(crate) mod core;
pub mod implementations;

// chordal decomposition only if SDPs are enabled
#[cfg(feature = "sdp")]
pub(crate) mod chordal;

//Here we expose only part of the solver internals
//and rearrange public modules a bit to give a more
//user friendly API

//allows declaration of cone constraints
pub use crate::solver::core::cones::{SupportedConeT, SupportedConeT::*};

//user facing traits required to interact with solver
pub use crate::solver::core::kktsolvers::LinearSolverInfo;
pub use crate::solver::core::{IPSolver, SolverStatus};

//user facing traits required to define new implementatiions
pub use crate::solver::core::traits;
pub use crate::solver::core::CoreSettings;

//user facing ffi interfaces
pub use crate::solver::core::ffi;

// read/write types if enabled
#[cfg(feature = "serde")]
pub use crate::solver::core::SolverJSONReadWrite;

//If we had implementations for multiple alternative
//problem formats, they would live here.   Since we
//only have default, it is exposed at the top level
//in the use statements directly below instead.

// pub mod implementations {
//     pub mod default {
//         pub use clarabel_solver::implementations::default::*;
//     }
// }

pub use crate::solver::implementations::default;
pub use crate::solver::implementations::default::*;
