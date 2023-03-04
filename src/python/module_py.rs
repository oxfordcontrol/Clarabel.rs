use super::*;
use pyo3::prelude::*;

// get/set for the solver's internal infinity limit
#[pyfunction(name = "get_infinity")]
fn get_infinity_py() -> f64 {
    crate::solver::get_infinity()
}
#[pyfunction(name = "set_infinity")]
fn set_infinity_py(v: f64) {
    crate::solver::set_infinity(v);
}
#[pyfunction(name = "default_infinity")]
fn default_infinity_py() {
    crate::solver::default_infinity();
}
// Python module and registry, which includes registration of the
// data types defined in the other files in this rust module
#[pymodule]
fn clarabel(_py: Python, m: &PyModule) -> PyResult<()> {
    //module version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    //module globs
    m.add_function(wrap_pyfunction!(get_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(set_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(default_infinity_py, m)?)
        .unwrap();

    // API Cone types
    m.add_class::<PyZeroConeT>()?;
    m.add_class::<PyNonnegativeConeT>()?;
    m.add_class::<PySecondOrderConeT>()?;
    m.add_class::<PyExponentialConeT>()?;
    m.add_class::<PyPowerConeT>()?;

    //other API data types
    m.add_class::<PySolverStatus>()?;
    m.add_class::<PyDefaultSolution>()?;
    m.add_class::<PyDefaultSettings>()?;

    // Main solver object
    m.add_class::<PyDefaultSolver>()?;

    Ok(())
}
