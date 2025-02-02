#![allow(non_snake_case)]

use crate::algebra::CscMatrix;
use core::ops::Deref;
use pyo3::prelude::*;

//We can't implement the foreign trait FromPyObject directly on CscMatrix
//since it is outside the crate, so put a dummy wrapper around it here.
#[pyclass]
pub struct PyCscMatrix(CscMatrix<f64>);

impl Deref for PyCscMatrix {
    type Target = CscMatrix<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<PyCscMatrix> for CscMatrix<f64> {
    fn from(mat: PyCscMatrix) -> Self {
        mat.0
    }
}

impl<'a> FromPyObject<'a> for PyCscMatrix {
    fn extract_bound(obj: &Bound<'a, pyo3::PyAny>) -> PyResult<Self> {
        let nzval: Vec<f64> = obj.getattr("data")?.extract()?;
        let rowval: Vec<usize> = obj.getattr("indices")?.extract()?;
        let colptr: Vec<usize> = obj.getattr("indptr")?.extract()?;
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;

        let mut mat = CscMatrix::new(shape[0], shape[1], colptr, rowval, nzval);

        // if the python object was not in standard format, force the rust
        // object to still be nicely formatted
        let is_canonical: bool = obj.getattr("has_canonical_format")?.extract()?;

        if !is_canonical {
            let _ = mat.canonicalize();
        }

        Ok(PyCscMatrix(mat))
    }
}
