use crate::*;
use pyo3::prelude::*;

/// Python module and registry, which includes registration of the
/// data types defined in the other files in this rust module
#[pymodule]
fn clarabel_python(_py: Python, m: &PyModule) -> PyResult<()> {
    // API Cone types
    m.add_class::<PyZeroConeT>()?;
    m.add_class::<PyNonnegativeConeT>()?;
    m.add_class::<PySecondOrderConeT>()?;

    //other API data types
    m.add_class::<PySolverStatus>()?;
    m.add_class::<PyDefaultSolveResult>()?;
    m.add_class::<PyDefaultSettings>()?;

    // Main solver object
    m.add_class::<PyDefaultSolver>()?;

    Ok(())
}
