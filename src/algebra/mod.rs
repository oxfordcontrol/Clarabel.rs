//! Clarabel algebra module.   
//!
//! __NB__: Users will not ordinarily need to interact with this crate except for defining
//! sparse matrix inputs in [`CscMatrix`](crate::algebra::CscMatrix) format.
//!
//! Clarabel comes with its own standalone implementation of all required internal algebraic operations implemented through the [`ScalarMath`](crate::algebra::VectorMath), [`VectorMath`](crate::algebra::VectorMath) and [`MatrixMath`](crate::algebra::MatrixMath) traits.   Future versions may add implementations of these traits through external libraries as optional features.
//!
//! All floating point calculations are represented internally on values implementing the
//! [`FloatT`](crate::algebra::FloatT) trait.

// first import and flatten the solver's collection
// of core numeric types and matrix / vector traits.

mod adjoint;
mod error_types;
mod floats;
mod math_traits;
mod matrix_traits;
mod matrix_types;
mod scalarmath;
mod symmetric;
mod vecmath;
pub use adjoint::*;
pub use error_types::*;
pub use floats::*;
pub use math_traits::*;
pub use matrix_traits::*;
pub use matrix_types::*;
pub use scalarmath::*;
pub use symmetric::*;
pub use vecmath::*;

// matrix implementations
mod csc;
mod dense;
mod densesym3x3;
pub use csc::*;
pub use dense::*;
pub(crate) use densesym3x3::*;

//mod densesym3x3;

//configure tests of internals
#[cfg(test)]
mod tests;
