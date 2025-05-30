//! Clarabel algebra module.   
//!
//! __NB__: Users will not ordinarily need to interact with this module except for defining
//! sparse matrix inputs in [`CscMatrix`] format.
//!
//! Clarabel comes with its own standalone implementation of all required internal algebraic operations implemented through the [`ScalarMath`], [`VectorMath`] and [`MatrixMath`] traits.   Future versions may add implementations of these traits through external libraries as optional features.
//!
//! All floating point calculations are represented internally on values implementing the
//! [`FloatT`] trait.

// first import and flatten the solver's collection
// of core numeric types and matrix / vector traits.

mod error_types;
mod floats;
mod math_traits;
mod matrix_traits;
mod matrix_types;
mod scalarmath;
mod utils;
mod vecmath;
pub use error_types::*;
pub use floats::*;
pub use math_traits::*;
pub use matrix_traits::*;
pub(crate) use matrix_types::*;
pub(crate) use scalarmath::*;
pub(crate) use utils::*;

// matrix implementations
mod csc;
pub use csc::*;

mod dense;
pub(crate) use dense::*;
// sparse vectors implementations (for chordal decomp only)
#[cfg(feature = "sdp")]
mod sparsevector;
#[cfg(feature = "sdp")]
pub(crate) use sparsevector::*;

//configure tests of internals
#[cfg(test)]
mod tests;
