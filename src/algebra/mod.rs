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
mod transcendental;
mod utils;
mod vecmath;
pub use error_types::*;
pub use floats::*;
pub use math_traits::*;
pub use matrix_traits::*;
pub(crate) use matrix_types::*;
pub(crate) use scalarmath::*;
pub use transcendental::{BitWidthDiagnostic, RealConst, RealSentinel, Transcendental};
pub(crate) use utils::*;

// exact-rational backend (feature-gated)
#[cfg(feature = "bigrational")]
mod rational;
#[cfg(feature = "bigrational")]
pub use rational::{
    arena_len, max_arena_bits, precision_bits, reset_arena, set_max_arena_bits,
    set_precision_bits, tighten_scalar, tighten_vec, with_max_arena_bits, with_precision,
    RationalReal,
};

// MPFR-backed float backend (feature-gated)
#[cfg(feature = "mpfr")]
mod mpfr;
#[cfg(feature = "mpfr")]
pub use mpfr::{
    default_precision as mpfr_default_precision, set_default_precision as set_mpfr_default_precision,
    with_mpfr_precision, MpfrFloat,
};

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
