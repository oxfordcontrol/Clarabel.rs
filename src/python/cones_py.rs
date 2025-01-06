#![allow(non_snake_case)]
#![allow(clippy::new_without_default)]

use crate::solver::core::{cones::SupportedConeT, cones::SupportedConeT::*};
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

// Python display functionality for genpowercone objects
// with floating point vector parameters
fn __repr__genpowcone(name: &str, alpha: &[f64], dim2: usize) -> String {
    let mut s = String::new();
    write!(s, "{}[\n    α = {:?},\n dim2 = {}\n]", name, alpha, dim2).unwrap();
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

#[pyclass(name = "GenPowerConeT")]
pub struct PyGenPowerConeT {
    #[pyo3(get)]
    pub α: Vec<f64>,
    #[pyo3(get)]
    pub dim2: usize,
}
#[pymethods]
impl PyGenPowerConeT {
    #[new]
    pub fn new(α: Vec<f64>, dim2: usize) -> Self {
        Self { α, dim2 }
    }
    pub fn __repr__(&self) -> String {
        __repr__genpowcone("GenPowerConeT", &self.α, self.dim2)
    }
}

#[pyclass(name = "PSDTriangleConeT")]
pub struct PyPSDTriangleConeT {
    #[pyo3(get)]
    pub dim: usize,
}
#[pymethods]
impl PyPSDTriangleConeT {
    #[new]
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn __repr__(&self) -> String {
        __repr__cone("PyPSDTriangleConeT", self.dim)
    }
}

// We can't implement the foreign trait FromPyObject directly on
// SupportedCone<f64> since both are defined outside the crate, so
// put a dummy wrapper around it here.

#[derive(Debug)]
pub struct PySupportedCone(SupportedConeT<f64>);

impl From<PySupportedCone> for SupportedConeT<f64> {
    fn from(cone: PySupportedCone) -> Self {
        cone.0
    }
}

impl<'a> FromPyObject<'a> for PySupportedCone {
    fn extract_bound(obj: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        let thetype = obj.get_type().name()?;
        let typestr = thetype.to_string_lossy();

        match typestr.as_ref() {
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
            "GenPowerConeT" => {
                let α: Vec<f64> = obj.getattr("α")?.extract()?;
                let dim2: usize = obj.getattr("dim2")?.extract()?;
                Ok(PySupportedCone(GenPowerConeT(α, dim2)))
            }
            "PSDTriangleConeT" => {
                let dim: usize = obj.getattr("dim")?.extract()?;
                Ok(PySupportedCone(PSDTriangleConeT(dim)))
            }
            _ => {
                let mut errmsg = String::new();
                write!(errmsg, "Unrecognized cone type : {}", thetype).unwrap();
                Err(PyTypeError::new_err(errmsg))
            }
        }
    }
}

pub(crate) fn _py_to_native_cones(cones: Vec<PySupportedCone>) -> Vec<SupportedConeT<f64>> {
    //force a vector of PySupportedCone back into a vector
    //of rust native SupportedCone.
    let mut out = Vec::with_capacity(cones.len());
    for cone in cones {
        out.push(cone.into());
    }
    out
}
