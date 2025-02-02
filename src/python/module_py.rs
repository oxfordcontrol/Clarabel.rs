use super::*;
use pyo3::prelude::*;

#[pyfunction(name = "force_load_blas_lapack")]
fn force_load_blas_lapack_py() {
    //force BLAS/LAPACK fcn pointer load
    //when using scipy lapack/blas
    #[cfg(sdp_pyblas)]
    crate::python::pyblas::force_load();
}

// get/set for the solver's internal infinity limit
#[pyfunction(name = "get_infinity")]
fn get_infinity_py() -> f64 {
    crate::get_infinity()
}
#[pyfunction(name = "set_infinity")]
fn set_infinity_py(v: f64) {
    crate::set_infinity(v);
}
#[pyfunction(name = "default_infinity")]
fn default_infinity_py() {
    crate::default_infinity();
}
#[pyfunction(name = "buildinfo")]
fn buildinfo_py() {
    crate::buildinfo();
}

// Python module and registry, which includes registration of the
// data types defined in the other files in this rust module
#[pymodule]
fn clarabel(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    //module version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // module initializer, called on module import
    m.add_function(wrap_pyfunction!(force_load_blas_lapack_py, m)?)
        .unwrap();

    //module globs
    m.add_function(wrap_pyfunction!(get_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(set_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(default_infinity_py, m)?)
        .unwrap();
    m.add_function(wrap_pyfunction!(buildinfo_py, m)?).unwrap();
    m.add_function(wrap_pyfunction!(read_from_file_py, m)?)
        .unwrap();

    // API Cone types
    m.add_class::<PyZeroConeT>()?;
    m.add_class::<PyNonnegativeConeT>()?;
    m.add_class::<PySecondOrderConeT>()?;
    m.add_class::<PyExponentialConeT>()?;
    m.add_class::<PyPowerConeT>()?;
    m.add_class::<PyGenPowerConeT>()?;
    m.add_class::<PyPSDTriangleConeT>()?;

    //other API data types
    m.add_class::<PySolverStatus>()?;
    m.add_class::<PyDefaultSolution>()?;
    m.add_class::<PyDefaultSettings>()?;

    // Main solver object
    m.add_class::<PyDefaultSolver>()?;

    Ok(())
}
