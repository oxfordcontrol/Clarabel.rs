use super::*;
use pyo3::prelude::*;

/// Python module and registry, which includes registration of the
/// data types defined in the other files in this rust module
#[pymodule]
fn clarabel(_py: Python, m: &PyModule) -> PyResult<()> {
    //module version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

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
