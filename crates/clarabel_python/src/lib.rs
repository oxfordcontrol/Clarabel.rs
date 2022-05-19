
#![allow(unused)]
#![allow(non_snake_case)]
fn main() {
use clarabel_solver as clarabel;
use clarabel_algebra as algebra;

use pyo3::prelude::*;
use pyo3::PyClass;
use rand::Rng;
use std::cmp::Ordering;
use std::io;
use core::ops::Deref;

// PJG: probably need better re-export here so only clarabel::* suffices.
use clarabel::*;
use clarabel::SupportedCones::*;
use clarabel::settings::*;
use clarabel::solver::*;
use clarabel::default::*;
use algebra::*;

//We can't implement a foreign trait directly on CscMatrix since 
//it is outside the crate, so put a dummy wrapper around it here.
struct CscMatrixPy(CscMatrix<f64>);

impl Deref for CscMatrixPy {
    type Target = CscMatrix<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> FromPyObject<'a> for CscMatrixPy {

    fn extract(obj: &'a PyAny) -> PyResult<Self> {

        let nzval : Vec<f64> = obj.getattr("data")?.extract()?;
        let indices : Vec<usize> = obj.getattr("indices")?.extract()?;
        let indptr : Vec<usize> = obj.getattr("indptr")?.extract()?;
        let nnz : usize = obj.getattr("nnz")?.extract()?;
        let shape : Vec<usize> = obj.getattr("shape")?.extract()?;

        let mat = CscMatrix{
            m : shape[0], 
            n : shape[1],
            colptr : indptr,
            rowval : indices,
            nzval,
        };

        Ok(CscMatrixPy(mat))
    }
}

#[pyfunction]
#[pyo3(name = "solve")]
fn solve_py(P: CscMatrixPy, q: Vec<f64>, A: CscMatrixPy, b: Vec<f64>) {

    let cone_types = [NonnegativeConeT];

    let cone_dims  = [b.len()];

    let settings = SettingsBuilder::default()
            .equilibrate_enable(true)
            .max_iter(50)
            .verbose(true)
            .build().unwrap();

    //PJG: no borrow on settings sucks here
    let mut solver = DefaultSolver::
        new(&P,&q,&A,&b,&cone_types,&cone_dims, settings);

    solver.solve();
}

 
/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn clarabel_python(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(solve_py, m)?)?;
    Ok(())
}

}
