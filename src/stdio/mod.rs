#![allow(unused_imports)]

// configure python specific stdout and stdin streams
// when compiled with the python feature.   This avoids
// problems when running within python notebooks etc.
#[cfg(feature = "python")]
pub(crate) use crate::python::io::{stderr, stdout};

#[cfg(not(feature = "python"))]
pub(crate) use std::io::{stderr, stdout};
