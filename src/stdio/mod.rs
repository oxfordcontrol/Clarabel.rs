#[cfg(not(feature = "python"))]
#[allow(unused_imports)]
pub(crate) use std::io::{stderr, stdout};

// configure python specific stdout and stdin strams
// when compiled with the python feature.   This avoids
// problems when running within python notebooks etc.
#[cfg(feature = "python")]
#[allow(unused_imports)]
pub(crate) use crate::python::io::{stderr, stdout};
