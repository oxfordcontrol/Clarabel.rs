//Rust hates greek characters
#![allow(confusable_idents)]

//package together all of the following and re-export
//in a flat structure :
// : cone traits and component implementation
// : kkt solver engines 
// : user settings 
// : core solver implementation and types 
// : trait definitions for core components

pub mod cones;
pub mod kktsolvers;
pub mod settings;
pub mod solver;
pub mod components;

//re-export everything in a flat structure
pub use cones::*;
pub use kktsolvers::*;
pub use settings::*;
pub use solver::*;
pub use components::*;

// include the default concrete implementation
pub mod implementations;
pub use implementations::default::*;


