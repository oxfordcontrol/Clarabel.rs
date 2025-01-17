#[cfg(not(feature = "python"))]
#[allow(unused_imports)]
pub(crate) use std::io::{stderr, stdout, Stdout};

// configure python specific stdout and stdin streams
// when compiled with the python feature.   This avoids
// problems when running within python notebooks etc.
#[cfg(feature = "python")]
#[allow(unused_imports)]
pub(crate) use crate::python::io::{stderr, stdout, Stdout};
