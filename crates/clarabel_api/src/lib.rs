//this module exports a reorganized module
//structure that defines a of set user-facing 
//solver API functions and types

pub mod core {

    //user facing algebra functions
    pub use clarabel_algebra::{CscMatrix};

    //allows declaration of cone constraints
    pub use clarabel_solver::core::cones::SupportedCones::*;

    //allows definitions of solver settings
    pub use clarabel_solver::core::{Settings,SettingsBuilder};

    //user facing traits required to interact with solver 
    pub use clarabel_solver::core::IPSolver;
}

pub mod implementations {

    //The default solver implementation and its 
    //user facing components
    pub mod default{
        pub use clarabel_solver::implementations::default::DefaultSolver;
    }
}

