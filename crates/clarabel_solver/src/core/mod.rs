//package together all of the following and re-export
//in a partially flattened structure :
// : core component traits
// : cone traits and standard cone implementations
// : kkt solver engines 
// : user settings 
// : main solver implementation 

pub mod traits;
pub mod cones;
pub mod kktsolvers;


//partially flatten top level pieces
mod solver;
mod settings;
pub use solver::*;
pub use settings::*;