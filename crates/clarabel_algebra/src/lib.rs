
// first import and flatten the solver's collection
// of core numeric types and matrix / vector traits.

mod floats;
mod traits;
mod cscmatrix;
pub use floats::*;
pub use traits::*;
pub use cscmatrix::*;


// here we select the particular numeric implementation of
// the core traits.  For now, we only have the hand-written
// one, so there is nothing to configure 
mod native;
pub use native::*;


