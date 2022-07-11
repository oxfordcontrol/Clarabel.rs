#![allow(unused)]
#![allow(non_snake_case)]

use core::ops::Deref;
use pyo3::{prelude::*, PyClass};
use pyo3::exceptions::PyTypeError;
use std::{cmp::Ordering, io, fmt::Write};

//python interface require some access to solver internals,
//so just use the internal crate definitions instead of the API.
use clarabel_algebra::CscMatrix;
use clarabel_solver as solver;
use solver::core::{cones::SupportedCones::*, 
    cones::SupportedCones,
    IPSolver, Settings, SettingsBuilder};
use solver::implementations::default::*;
use crate::*;


fn _py_to_native_cones(cones: Vec<PySupportedCones>) -> Vec<SupportedCones<f64>> {

        //force a vector of PySupportedCones back into a vector
        //of rust native SupportedCones.  The Py cone is just 
        //a wrapper; deref gives us the native object.
        cones.iter()
             .map(|x| *x.deref())
             .collect()

}

#[pyfunction]
#[pyo3(name = "solve")]
fn py_solve(P: PyCscMatrix, 
            q: Vec<f64>, 
            A: PyCscMatrix, 
            b: Vec<f64>, 
            cones: Vec<PySupportedCones>) {

    let cones = _py_to_native_cones(cones);

    let settings = SettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .verbose(true)
        .build()
        .unwrap();

    //PJG: no borrow on settings sucks here
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();
}


/// Python module and registry, which includes registration of the 
/// data types defined in the other files in this rust module
#[pymodule]
fn clarabel_python(_py: Python, m: &PyModule) -> PyResult<()> {

    // API Cone types
    m.add_class::<PyZeroConeT>()?;
    m.add_class::<PyNonnegativeConeT>()?;
    m.add_class::<PySecondOrderConeT>()?;

    // Clarabel API functions 
    m.add_function(wrap_pyfunction!(py_solve, m)?)?;

    Ok(())
}
