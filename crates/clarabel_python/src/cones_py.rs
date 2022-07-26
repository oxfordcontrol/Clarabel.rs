#![allow(non_snake_case)]

use core::ops::Deref;
use pyo3::{exceptions::PyTypeError, prelude::*};
use std::fmt::Write;

// python interface require some access to solver internals,
// so just use the internal crate definitions instead of the API.
use clarabel_solver::core::{cones::SupportedCones, cones::SupportedCones::*};

// generic Python display functionality for cone objects
fn __repr__cone(name: &str, dim: usize) -> String {
    let mut s = String::new();
    write!(s, "{}({})", name, dim).unwrap();
    s
}

#[pyclass(name = "ZeroConeT")]
pub struct PyZeroConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PyZeroConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("ZeroConeT", self.dim)
    }
}

#[pyclass(name = "NonnegativeConeT")]
pub struct PyNonnegativeConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PyNonnegativeConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("NonnegativeConeT", self.dim)
    }
}

#[pyclass(name = "SecondOrderConeT")]
pub struct PySecondOrderConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PySecondOrderConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("SecondOrderConeT", self.dim)
    }
}

// We can't implement the foreign trait FromPyObject directly on
// SupportedCones<f64> since both are defined outside the crate, so
// put a dummy wrapper around it here.

#[derive(Debug)]
pub struct PySupportedCones(SupportedCones<f64>);

impl Deref for PySupportedCones {
    type Target = SupportedCones<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromPyObject<'a> for PySupportedCones {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let thetype = obj.get_type().name()?;

        match thetype {
            "ZeroConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCones(ZeroConeT(dim)))
            }
            "NonnegativeConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCones(NonnegativeConeT(dim)))
            }
            "SecondOrderConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCones(SecondOrderConeT(dim)))
            }
            _ => {
                let mut errmsg = String::new();
                write!(errmsg, "Unrecognized cone type : {}", thetype).unwrap();
                Err(PyTypeError::new_err(errmsg))
            }
        }
    }
}

pub(crate) fn _py_to_native_cones(cones: Vec<PySupportedCones>) -> Vec<SupportedCones<f64>> {
    //force a vector of PySupportedCones back into a vector
    //of rust native SupportedCones.  The Py cone is just
    //a wrapper; deref gives us the native object.
    cones.iter().map(|x| *x.deref()).collect()
}
