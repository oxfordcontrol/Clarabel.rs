//this module exports a reorganized module
//structure that defines a of set user-facing
//solver API functions and types

pub mod algebra {
    pub use clarabel_algebra::*;
}

pub mod qdldl {
    pub use clarabel_qdldl::*;
}

pub mod timers {
    pub use clarabel_timers::*;
}

pub mod solver {

    //Here we expose only part of the solver internals
    //and rearrange modules a bit to give a more user
    //friendly API

    //allows declaration of cone constraints
    pub use clarabel_solver::core::cones::{SupportedCones, SupportedCones::*};

    //user facing traits required to interact with solver
    pub use clarabel_solver::core::{IPSolver, SolverStatus};

    //If we have implemtations for multple alternative
    //problem formats, they would live here.   Since we
    //only have default, it is exposed at the top level
    //in the use statements directly below instead.

    // pub mod implementations {
    //     pub mod default {
    //         pub use clarabel_solver::implementations::default::*;
    //     }
    // }

    pub use clarabel_solver::implementations::default::*;
}
