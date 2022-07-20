#![allow(unused)]
#![allow(non_snake_case)]

use core::ops::Deref;
use pyo3::exceptions::PyTypeError;
use pyo3::{prelude::*, PyClass};
use std::{cmp::Ordering, fmt::Write, io};

//python interface require some access to solver internals,
//so just use the internal crate definitions instead of the API.
use clarabel_algebra::CscMatrix;

//We can't implement the foreign trait FromPyObject directly on CscMatrix
//since it is outside the crate, so put a dummy wrapper around it here.
pub struct PyCscMatrix(CscMatrix<f64>);

impl Deref for PyCscMatrix {
    type Target = CscMatrix<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromPyObject<'a> for PyCscMatrix {
    fn extract(obj: &'a PyAny) -> PyResult<Self> {
        let nzval: Vec<f64> = obj.getattr("data")?.extract()?;
        let indices: Vec<usize> = obj.getattr("indices")?.extract()?;
        let indptr: Vec<usize> = obj.getattr("indptr")?.extract()?;
        let nnz: usize = obj.getattr("nnz")?.extract()?;
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;

        let mat = CscMatrix {
            m: shape[0],
            n: shape[1],
            colptr: indptr,
            rowval: indices,
            nzval,
        };

        Ok(PyCscMatrix(mat))
    }
}
