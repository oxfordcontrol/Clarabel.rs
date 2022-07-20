//package together all of the following and re-export
//in a partially flattened structure :
// : core component traits
// : cone traits and standard cone implementations
// : kkt solver engines
// : user settings
// : main solver implementation

pub mod cones;
pub mod kktsolvers;
pub mod traits;

//partially flatten top level pieces
mod settings;
mod solver;
pub use settings::*;
pub use solver::*;
