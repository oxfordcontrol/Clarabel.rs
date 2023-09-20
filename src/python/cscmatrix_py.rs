#![allow(unused)]
#![allow(non_snake_case)]

use crate::algebra::CscMatrix;
use core::ops::Deref;
use pyo3::exceptions::PyTypeError;
use pyo3::{prelude::*, PyClass};
use std::{cmp::Ordering, fmt::Write, io};

//We can't implement the foreign trait FromPyObject directly on CscMatrix
//since it is outside the crate, so put a dummy wrapper around it here.
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
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let nzval: Vec<f64> = obj.getattr("data")?.extract()?;
        let rowval: Vec<usize> = obj.getattr("indices")?.extract()?;
        let colptr: Vec<usize> = obj.getattr("indptr")?.extract()?;
        let nnz: usize = obj.getattr("nnz")?.extract()?;
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;

        let mat = CscMatrix::new(shape[0], shape[1], colptr, rowval, nzval);

        Ok(PyCscMatrix(mat))
    }
}
