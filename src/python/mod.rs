//! Clarabel Python interface.
//!
//! This module implements a wrapper for the Rust version of Python using
//! [PyO3](https://pyo3.rs/).   To build these wrappers from `cargo`, compile the crate with
//! `--features python`.   This module has no public API.
//!
//! It should not normally be necessary to compile the Python wrapper from
//! source.   See the [Python Installation Documentation](https://oxfordcontrol.github.io/ClarabelDocs/stable/python/installation_py/).
//!

mod cones_py;
mod cscmatrix_py;
mod impl_default_py;
pub(crate) mod io;
mod module_py;

// compile this module if no local blas/lapack library
// has been specified, and we want to use the python/scipy
// version instead.  sdp_pyblas is defined in build.rs
#[cfg(sdp_pyblas)]
pub(crate) mod pyblas;

// NB : Nothing is actually public here, but the python module itself
// is made public so that we can force the docstring above to appear
// in the API documentation and give the link.

pub(crate) use cones_py::*;
pub(crate) use cscmatrix_py::*;
pub(crate) use impl_default_py::*;
