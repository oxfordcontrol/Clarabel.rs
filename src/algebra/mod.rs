//! Clarabel algebra module.   
//!
//! __NB__: Users will not ordinarily need to interact with this crate except for defining
//! sparse matrix inputs in [CscMatrix](crate::algebra::CscMatrix) format.
//!
//! Clarabel comes with its own standalone implementation of all required internal algebraic
//! operations implemented through the [ScalarMath](crate::algebra::VectorMath),
//! [VectorMath](crate::algebra::VectorMath) and [MatrixMath](crate::algebra::MatrixMath)  
//! traits.  Future versions may add implementations of these traits through external libraries
//! as optional features.
//!
//! All floating point calculations are represented internally on values implementing the
//! [FloatT](crate::algebra::FloatT) trait.

// first import and flatten the solver's collection
// of core numeric types and matrix / vector traits.

mod floats;
mod matrix_types;
mod matrix_utils;
mod traits;
pub use floats::*;
pub use matrix_types::*;
pub use matrix_utils::*;
pub use traits::*;

// here we select the particular numeric implementation of
// the core traits.  For now, we only have the hand-written
// one, so there is nothing to configure
mod native;
pub use native::*;

//configure tests of internals
#[cfg(test)]
mod tests;
