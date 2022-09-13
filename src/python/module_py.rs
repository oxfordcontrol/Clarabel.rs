use super::*;
use pyo3::prelude::*;

#[pyfunction(name = "__version__")]
fn version_py() -> String {
    crate::VERSION.to_string()
}

/// Python module and registry, which includes registration of the
/// data types defined in the other files in this rust module
#[pymodule]
fn clarabel(_py: Python, m: &PyModule) -> PyResult<()> {
    //module version
    m.add_function(wrap_pyfunction!(version_py, m)?).unwrap();

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
