// Python wrappers and interface for the Default solver
// implementation and its related types.

#![allow(non_snake_case)]

use super::*;
use crate::{
    algebra::CscMatrix,
    solver::{
        core::{
            traits::{InfoPrint, Settings},
            IPSolver, SolverStatus,
        },
        implementations::default::*,
        SolverJSONReadWrite,
    },
};
use pyo3::{exceptions::PyException, prelude::*, types::PyDict};
use std::fmt::Write;

//Here we end up repeating several datatypes defined internally
//in the Clarabel default implementation.   We would prefer
//to just apply the PyO3 macros to autoderive these types,
//except there are currently problems using cfg_attr with
//the PyO3 get/set attribute.  Pyo3 also does not seem to
//support autoderivation of python types from Rust structs
//that use generics.   See here:
//
// https://github.com/PyO3/pyo3/issues/780
// https://github.com/PyO3/pyo3/issues/1003
// https://github.com/PyO3/pyo3/issues/1088

// ----------------------------------
// DefaultSolution
// ----------------------------------

#[derive(Debug)]
#[pyclass(name = "DefaultSolution")]
pub struct PyDefaultSolution {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub s: Vec<f64>,
    #[pyo3(get)]
    pub z: Vec<f64>,
    #[pyo3(get)]
    pub status: PySolverStatus,
    #[pyo3(get)]
    pub obj_val: f64,
    #[pyo3(get)]
    pub obj_val_dual: f64,
    #[pyo3(get)]
    pub solve_time: f64,
    #[pyo3(get)]
    pub iterations: u32,
    #[pyo3(get)]
    pub r_prim: f64,
    #[pyo3(get)]
    pub r_dual: f64,
}

impl PyDefaultSolution {
    pub(crate) fn new_from_internal(result: &DefaultSolution<f64>) -> Self {
        let x = result.x.clone();
        let s = result.s.clone();
        let z = result.z.clone();
        let status = PySolverStatus::new_from_internal(&result.status);
        Self {
            x,
            s,
            z,
            obj_val: result.obj_val,
            obj_val_dual: result.obj_val_dual,
            status,
            solve_time: result.solve_time,
            iterations: result.iterations,
            r_prim: result.r_prim,
            r_dual: result.r_dual,
        }
    }
}

#[pymethods]
impl PyDefaultSolution {
    pub fn __repr__(&self) -> String {
        "Clarabel solution object".to_string()
    }
}

// ----------------------------------
// Solver Status
// ----------------------------------

#[derive(PartialEq, Debug, Clone, Copy)]
#[pyclass(eq, eq_int, name = "SolverStatus")]
pub enum PySolverStatus {
    Unsolved = 0,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    AlmostSolved,
    AlmostPrimalInfeasible,
    AlmostDualInfeasible,
    MaxIterations,
    MaxTime,
    NumericalError,
    InsufficientProgress,
}

impl PySolverStatus {
    pub(crate) fn new_from_internal(status: &SolverStatus) -> Self {
        match status {
            SolverStatus::Unsolved => PySolverStatus::Unsolved,
            SolverStatus::Solved => PySolverStatus::Solved,
            SolverStatus::PrimalInfeasible => PySolverStatus::PrimalInfeasible,
            SolverStatus::DualInfeasible => PySolverStatus::DualInfeasible,
            SolverStatus::AlmostSolved => PySolverStatus::AlmostSolved,
            SolverStatus::AlmostPrimalInfeasible => PySolverStatus::AlmostPrimalInfeasible,
            SolverStatus::AlmostDualInfeasible => PySolverStatus::AlmostDualInfeasible,
            SolverStatus::MaxIterations => PySolverStatus::MaxIterations,
            SolverStatus::MaxTime => PySolverStatus::MaxTime,
            SolverStatus::NumericalError => PySolverStatus::NumericalError,
            SolverStatus::InsufficientProgress => PySolverStatus::InsufficientProgress,
        }
    }
}

#[pymethods]
impl PySolverStatus {
    pub fn __repr__(&self) -> String {
        match self {
            PySolverStatus::Unsolved => "Unsolved",
            PySolverStatus::Solved => "Solved",
            PySolverStatus::PrimalInfeasible => "PrimalInfeasible",
            PySolverStatus::DualInfeasible => "DualInfeasible",
            PySolverStatus::AlmostSolved => "AlmostSolved",
            PySolverStatus::AlmostPrimalInfeasible => "AlmostPrimalInfeasible",
            PySolverStatus::AlmostDualInfeasible => "AlmostDualInfeasible",
            PySolverStatus::MaxIterations => "MaxIterations",
            PySolverStatus::MaxTime => "MaxTime",
            PySolverStatus::NumericalError => "NumericalError",
            PySolverStatus::InsufficientProgress => "InsufficientProgress",
        }
        .to_string()
    }

    // mapping of solver status to CVXPY keys is done via a hash
    pub fn __hash__(&self) -> u32 {
        *self as u32
    }
}

// ----------------------------------
// Solver Settings
// ----------------------------------

#[derive(Debug, Clone)]
#[pyclass(name = "DefaultSettings")]
pub struct PyDefaultSettings {
    #[pyo3(get, set)]
    pub max_iter: u32,
    #[pyo3(get, set)]
    pub time_limit: f64,
    #[pyo3(get, set)]
    pub verbose: bool,
    #[pyo3(get, set)]
    pub max_step_fraction: f64,

    //full accuracy solution tolerances
    #[pyo3(get, set)]
    pub tol_gap_abs: f64,
    #[pyo3(get, set)]
    pub tol_gap_rel: f64,
    #[pyo3(get, set)]
    pub tol_feas: f64,
    #[pyo3(get, set)]
    pub tol_infeas_abs: f64,
    #[pyo3(get, set)]
    pub tol_infeas_rel: f64,
    #[pyo3(get, set)]
    pub tol_ktratio: f64,

    //reduced accuracy solution tolerances
    #[pyo3(get, set)]
    pub reduced_tol_gap_abs: f64,
    #[pyo3(get, set)]
    pub reduced_tol_gap_rel: f64,
    #[pyo3(get, set)]
    pub reduced_tol_feas: f64,
    #[pyo3(get, set)]
    pub reduced_tol_infeas_abs: f64,
    #[pyo3(get, set)]
    pub reduced_tol_infeas_rel: f64,
    #[pyo3(get, set)]
    pub reduced_tol_ktratio: f64,

    // data equilibration
    #[pyo3(get, set)]
    pub equilibrate_enable: bool,
    #[pyo3(get, set)]
    pub equilibrate_max_iter: u32,
    #[pyo3(get, set)]
    pub equilibrate_min_scaling: f64,
    #[pyo3(get, set)]
    pub equilibrate_max_scaling: f64,

    //step size settings
    #[pyo3(get, set)]
    pub linesearch_backtrack_step: f64,
    #[pyo3(get, set)]
    pub min_switch_step_length: f64,
    #[pyo3(get, set)]
    pub min_terminate_step_length: f64,

    // KKT settings
    #[pyo3(get, set)]
    pub max_threads: u32,
    #[pyo3(get, set)]
    pub direct_kkt_solver: bool,
    #[pyo3(get, set)]
    pub direct_solve_method: String,

    // static regularization parameters
    #[pyo3(get, set)]
    pub static_regularization_enable: bool,
    #[pyo3(get, set)]
    pub static_regularization_constant: f64,
    #[pyo3(get, set)]
    pub static_regularization_proportional: f64,

    // dynamic regularization parameters
    #[pyo3(get, set)]
    pub dynamic_regularization_enable: bool,
    #[pyo3(get, set)]
    pub dynamic_regularization_eps: f64,
    #[pyo3(get, set)]
    pub dynamic_regularization_delta: f64,

    // iterative refinement (for QDLDL)
    #[pyo3(get, set)]
    pub iterative_refinement_enable: bool,
    #[pyo3(get, set)]
    pub iterative_refinement_reltol: f64,
    #[pyo3(get, set)]
    pub iterative_refinement_abstol: f64,
    #[pyo3(get, set)]
    pub iterative_refinement_max_iter: u32,
    #[pyo3(get, set)]
    pub iterative_refinement_stop_ratio: f64,

    // preprocessing
    #[pyo3(get, set)]
    pub presolve_enable: bool,

    //chordal decomposition (python must be built with "sdp" feature)
    #[pyo3(get, set)]
    pub chordal_decomposition_enable: bool,
    #[pyo3(get, set)]
    pub chordal_decomposition_merge_method: String,
    #[pyo3(get, set)]
    pub chordal_decomposition_compact: bool,
    #[pyo3(get, set)]
    pub chordal_decomposition_complete_dual: bool,
}

#[pymethods]
impl PyDefaultSettings {
    #[new]
    pub fn new() -> Self {
        PyDefaultSettings::new_from_internal(&DefaultSettings::<f64>::default())
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    pub fn py_default() -> Self {
        PyDefaultSettings::default()
    }

    pub fn __repr__(&self) -> String {
        let mut s = String::new();
        write!(s, "{:#?}", self).unwrap();
        s
    }
}

//Default not really necessary, but keeps clippy happy....
impl Default for PyDefaultSettings {
    fn default() -> Self {
        PyDefaultSettings::new()
    }
}

impl PyDefaultSettings {
    pub(crate) fn new_from_internal(set: &DefaultSettings<f64>) -> Self {
        PyDefaultSettings {
            max_iter: set.max_iter,
            time_limit: set.time_limit,
            verbose: set.verbose,
            tol_gap_abs: set.tol_gap_abs,
            tol_gap_rel: set.tol_gap_rel,
            tol_feas: set.tol_feas,
            tol_infeas_abs: set.tol_infeas_abs,
            tol_infeas_rel: set.tol_infeas_rel,
            tol_ktratio: set.tol_ktratio,
            reduced_tol_gap_abs: set.reduced_tol_gap_abs,
            reduced_tol_gap_rel: set.reduced_tol_gap_rel,
            reduced_tol_feas: set.reduced_tol_feas,
            reduced_tol_infeas_abs: set.reduced_tol_infeas_abs,
            reduced_tol_infeas_rel: set.reduced_tol_infeas_rel,
            reduced_tol_ktratio: set.reduced_tol_ktratio,
            max_step_fraction: set.max_step_fraction,
            equilibrate_enable: set.equilibrate_enable,
            equilibrate_max_iter: set.equilibrate_max_iter,
            equilibrate_min_scaling: set.equilibrate_min_scaling,
            equilibrate_max_scaling: set.equilibrate_max_scaling,
            linesearch_backtrack_step: set.linesearch_backtrack_step,
            min_switch_step_length: set.min_switch_step_length,
            min_terminate_step_length: set.min_terminate_step_length,
            max_threads: set.max_threads,
            direct_kkt_solver: set.direct_kkt_solver,
            direct_solve_method: set.direct_solve_method.clone(),
            static_regularization_enable: set.static_regularization_enable,
            static_regularization_constant: set.static_regularization_constant,
            static_regularization_proportional: set.static_regularization_proportional,
            dynamic_regularization_enable: set.dynamic_regularization_enable,
            dynamic_regularization_eps: set.dynamic_regularization_eps,
            dynamic_regularization_delta: set.dynamic_regularization_delta,
            iterative_refinement_enable: set.iterative_refinement_enable,
            iterative_refinement_reltol: set.iterative_refinement_reltol,
            iterative_refinement_abstol: set.iterative_refinement_abstol,
            iterative_refinement_max_iter: set.iterative_refinement_max_iter,
            iterative_refinement_stop_ratio: set.iterative_refinement_stop_ratio,
            presolve_enable: set.presolve_enable,
            chordal_decomposition_enable: set.chordal_decomposition_enable,
            chordal_decomposition_merge_method: set.chordal_decomposition_merge_method.clone(),
            chordal_decomposition_compact: set.chordal_decomposition_compact,
            chordal_decomposition_complete_dual: set.chordal_decomposition_complete_dual,
        }
    }

    pub(crate) fn to_internal(&self) -> Result<DefaultSettings<f64>, PyErr> {
        // convert python settings -> Rust

        let settings = DefaultSettings::<f64> {
            max_iter: self.max_iter,
            time_limit: self.time_limit,
            verbose: self.verbose,
            tol_gap_abs: self.tol_gap_abs,
            tol_gap_rel: self.tol_gap_rel,
            tol_feas: self.tol_feas,
            tol_infeas_abs: self.tol_infeas_abs,
            tol_infeas_rel: self.tol_infeas_rel,
            tol_ktratio: self.tol_ktratio,
            reduced_tol_gap_abs: self.reduced_tol_gap_abs,
            reduced_tol_gap_rel: self.reduced_tol_gap_rel,
            reduced_tol_feas: self.reduced_tol_feas,
            reduced_tol_infeas_abs: self.reduced_tol_infeas_abs,
            reduced_tol_infeas_rel: self.reduced_tol_infeas_rel,
            reduced_tol_ktratio: self.reduced_tol_ktratio,
            max_step_fraction: self.max_step_fraction,
            equilibrate_enable: self.equilibrate_enable,
            equilibrate_max_iter: self.equilibrate_max_iter,
            equilibrate_min_scaling: self.equilibrate_min_scaling,
            equilibrate_max_scaling: self.equilibrate_max_scaling,
            linesearch_backtrack_step: self.linesearch_backtrack_step,
            min_switch_step_length: self.min_switch_step_length,
            min_terminate_step_length: self.min_terminate_step_length,
            max_threads: self.max_threads,
            direct_kkt_solver: self.direct_kkt_solver,
            direct_solve_method: self.direct_solve_method.clone(),
            static_regularization_enable: self.static_regularization_enable,
            static_regularization_constant: self.static_regularization_constant,
            static_regularization_proportional: self.static_regularization_proportional,
            dynamic_regularization_enable: self.dynamic_regularization_enable,
            dynamic_regularization_eps: self.dynamic_regularization_eps,
            dynamic_regularization_delta: self.dynamic_regularization_delta,
            iterative_refinement_enable: self.iterative_refinement_enable,
            iterative_refinement_reltol: self.iterative_refinement_reltol,
            iterative_refinement_abstol: self.iterative_refinement_abstol,
            iterative_refinement_max_iter: self.iterative_refinement_max_iter,
            iterative_refinement_stop_ratio: self.iterative_refinement_stop_ratio,
            presolve_enable: self.presolve_enable,
            chordal_decomposition_enable: self.chordal_decomposition_enable,
            chordal_decomposition_merge_method: self.chordal_decomposition_merge_method.clone(),
            chordal_decomposition_compact: self.chordal_decomposition_compact,
            chordal_decomposition_complete_dual: self.chordal_decomposition_complete_dual,
        };

        //manually validate settings from Python side
        match settings.validate() {
            Ok(_) => Ok(settings),
            Err(e) => Err(PyException::new_err(format!("Invalid settings: {}", e))),
        }
    }
}

// ----------------------------------
// Solver
// ----------------------------------
impl From<DataUpdateError> for PyErr {
    fn from(err: DataUpdateError) -> Self {
        PyException::new_err(err.to_string())
    }
}

#[pyclass(name = "DefaultSolver")]
pub struct PyDefaultSolver {
    inner: DefaultSolver<f64>,
}

#[pymethods]
impl PyDefaultSolver {
    #[new]
    fn new(
        P: PyCscMatrix,
        q: Vec<f64>,
        A: PyCscMatrix,
        b: Vec<f64>,
        cones: Vec<PySupportedCone>,
        settings: PyDefaultSettings,
    ) -> PyResult<Self> {
        let cones = _py_to_native_cones(cones);
        let settings = settings.to_internal()?;

        let solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
        Ok(Self { inner: solver })
    }

    fn solve(&mut self, py: Python<'_>) -> PyDefaultSolution {
        py.allow_threads(|| self.inner.solve());
        PyDefaultSolution::new_from_internal(&self.inner.solution)
    }

    pub fn __repr__(&self) -> String {
        "Clarabel model with Float precision: f64".to_string()
    }

    fn print_configuration(&mut self) {
        // force a print of the configuration regardless
        // of the verbosity settings.   Save them here first.
        let verbose = self.inner.settings.core().verbose;

        self.inner.settings.core_mut().verbose = true;
        self.inner
            .info
            .print_configuration(&self.inner.settings, &self.inner.data, &self.inner.cones)
            .unwrap();

        // revert back to user option
        self.inner.settings.core_mut().verbose = verbose;
    }

    fn print_timers(&self) {
        match &self.inner.timers {
            Some(timers) => timers.print(),
            None => println!("no timers enabled"),
        };
    }

    fn write_to_file(&self, filename: &str) -> PyResult<()> {
        let mut file = std::fs::File::create(filename)?;
        self.inner.write_to_file(&mut file)?;
        Ok(())
    }

    #[pyo3(signature = (**kwds))]
    fn update(&mut self, kwds: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        for (key, value) in kwds.unwrap().iter() {
            let key = key.extract::<String>()?;

            match key.as_str() {
                "P" => match _py_to_matrix_update(value) {
                    Some(PyMatrixUpdateData::Matrix(M)) => {
                        self.inner.update_P(&M)?;
                    }
                    Some(PyMatrixUpdateData::Vector(v)) => {
                        self.inner.update_P(&v)?;
                    }
                    Some(PyMatrixUpdateData::Tuple((indices, values))) => {
                        self.inner.update_P(&(indices, values))?;
                    }
                    None => {
                        return Err(PyException::new_err("Invalid P update data"));
                    }
                },
                "A" => match _py_to_matrix_update(value) {
                    Some(PyMatrixUpdateData::Matrix(M)) => {
                        self.inner.update_A(&M)?;
                    }
                    Some(PyMatrixUpdateData::Vector(v)) => {
                        self.inner.update_A(&v)?;
                    }
                    Some(PyMatrixUpdateData::Tuple((indices, values))) => {
                        self.inner.update_A(&(indices, values))?;
                    }
                    None => {
                        return Err(PyException::new_err("Invalid A update data"));
                    }
                },
                "q" => match _py_to_vector_update(value) {
                    Some(PyVectorUpdateData::Vector(v)) => {
                        self.inner.update_q(&v)?;
                    }
                    Some(PyVectorUpdateData::Tuple((indices, values))) => {
                        self.inner.update_q(&(indices, values))?;
                    }
                    None => {
                        return Err(PyException::new_err("Invalid q update data"));
                    }
                },
                "b" => match _py_to_vector_update(value) {
                    Some(PyVectorUpdateData::Vector(v)) => {
                        self.inner.update_b(&v)?;
                    }
                    Some(PyVectorUpdateData::Tuple((indices, values))) => {
                        self.inner.update_b(&(indices, values))?;
                    }
                    None => {
                        return Err(PyException::new_err("Invalid b update data"));
                    }
                },
                "settings" => {
                    let settings: PyDefaultSettings = value.extract()?;
                    let settings = settings.to_internal()?;
                    self.inner.settings = settings;
                }
                _ => {
                    println!("unrecognized key: {}", key);
                }
            }
        }
        Ok(())
    }

    // return the currently configured settings of a solver.  If settings
    // are to be overridden, modify this object then pass back using kwargs
    // update(settings=settings)
    fn get_settings(&self) -> PyDefaultSettings {
        PyDefaultSettings::new_from_internal(&self.inner.settings)
    }

    fn is_data_update_allowed(&self) -> bool {
        self.inner.is_data_update_allowed()
    }
}

enum PyMatrixUpdateData {
    Matrix(CscMatrix<f64>),
    Vector(Vec<f64>),
    Tuple((Vec<usize>, Vec<f64>)),
}

enum PyVectorUpdateData {
    Vector(Vec<f64>),
    Tuple((Vec<usize>, Vec<f64>)),
}

impl From<PyVectorUpdateData> for PyMatrixUpdateData {
    fn from(val: PyVectorUpdateData) -> Self {
        match val {
            PyVectorUpdateData::Vector(v) => PyMatrixUpdateData::Vector(v),
            PyVectorUpdateData::Tuple((indices, values)) => {
                PyMatrixUpdateData::Tuple((indices, values))
            }
        }
    }
}

fn _py_to_matrix_update(arg: Bound<'_, PyAny>) -> Option<PyMatrixUpdateData> {
    // try converting to a csc matrix
    let csc: Option<CscMatrix<f64>> = arg.extract::<PyCscMatrix>().ok().map(|x| x.into());
    if let Some(csc) = csc {
        return Some(PyMatrixUpdateData::Matrix(csc));
    }

    // try as vector data
    _py_to_vector_update(arg).map(|x| x.into())
}

fn _py_to_vector_update(arg: Bound<'_, PyAny>) -> Option<PyVectorUpdateData> {
    // try converting to a complete data vector
    let values: Option<Vec<f64>> = arg.extract().ok();
    if let Some(values) = values {
        return Some(PyVectorUpdateData::Vector(values));
    }

    // try converting to a tuple of data and index vectors
    let tuple = arg.extract::<(Vec<usize>, Vec<f64>)>().ok();
    if let Some(tuple) = tuple {
        return Some(PyVectorUpdateData::Tuple(tuple));
    }
    None
}

#[pyfunction(name = "read_from_file")]
#[pyo3(signature = (filename, settings=None))]
pub fn read_from_file_py(
    filename: &str,
    settings: Option<PyDefaultSettings>,
) -> PyResult<PyDefaultSolver> {
    let mut file = std::fs::File::open(filename)?;

    match settings {
        Some(settings) => {
            let settings = settings.to_internal()?;
            let solver = DefaultSolver::<f64>::read_from_file(&mut file, Some(settings))?;
            Ok(PyDefaultSolver { inner: solver })
        }
        None => {
            let solver = DefaultSolver::<f64>::read_from_file(&mut file, None)?;
            Ok(PyDefaultSolver { inner: solver })
        }
    }
}
