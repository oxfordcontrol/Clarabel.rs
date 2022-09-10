#![allow(non_snake_case)]
#![allow(clippy::new_without_default)]

use crate::solver::core::{cones::SupportedCone, cones::SupportedCone::*};
use core::ops::Deref;
use pyo3::{exceptions::PyTypeError, prelude::*};
use std::fmt::Write;

// generic Python display functionality for cone objects
fn __repr__cone(name: &str, dim: usize) -> String {
    let mut s = String::new();
    write!(s, "{}({})", name, dim).unwrap();
    s
}

// generic Python display functionality for cone objects
// with no parameters, specifically 3d expcone
fn __repr__cone__noparams(name: &str) -> String {
    let mut s = String::new();
    write!(s, "{}()", name).unwrap();
    s
}

// generic Python display functionality for cone objects
// with floating point parameters (specifically power cone)
fn __repr__cone__float(name: &str, pow: f64) -> String {
    let mut s = String::new();
    write!(s, "{}({})", name, pow).unwrap();
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

#[pyclass(name = "ExponentialConeT")]
pub struct PyExponentialConeT {}
#[pymethods]
impl PyExponentialConeT {
    #[new]
    pub fn new() -> Self {
        Self {}
    }
    pub fn __repr__(&self) -> String {
        __repr__cone__noparams("ExponentialConeT")
    }
}

#[pyclass(name = "PowerConeT")]
pub struct PyPowerConeT {
    #[pyo3(get)]
    pub α: f64,
}
#[pymethods]
impl PyPowerConeT {
    #[new]
    pub fn new(α: f64) -> Self {
        Self { α }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone__float("PowerConeT", self.α)
    }
}

// We can't implement the foreign trait FromPyObject directly on
// SupportedCone<f64> since both are defined outside the crate, so
// put a dummy wrapper around it here.

#[derive(Debug)]
pub struct PySupportedCone(SupportedCone<f64>);

impl Deref for PySupportedCone {
    type Target = SupportedCone<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromPyObject<'a> for PySupportedCone {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let thetype = obj.get_type().name()?;

        match thetype {
            "ZeroConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(ZeroConeT(dim)))
            }
            "NonnegativeConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(NonnegativeConeT(dim)))
            }
            "SecondOrderConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(SecondOrderConeT(dim)))
            }
            "ExponentialConeT" => Ok(PySupportedCone(ExponentialConeT())),
            "PowerConeT" => {
                let α: f64 = obj.getattr("α")?.extract()?;
                Ok(PySupportedCone(PowerConeT(α)))
            }
            _ => {
                let mut errmsg = String::new();
                write!(errmsg, "Unrecognized cone type : {}", thetype).unwrap();
                Err(PyTypeError::new_err(errmsg))
            }
        }
    }
}

pub(crate) fn _py_to_native_cones(cones: Vec<PySupportedCone>) -> Vec<SupportedCone<f64>> {
    //force a vector of PySupportedCone back into a vector
    //of rust native SupportedCone.  The Py cone is just
    //a wrapper; deref gives us the native object.
    cones.iter().map(|x| *x.deref()).collect()
}
