#![allow(non_snake_case)]

use pyo3::prelude::*;
use std::fmt::Write;

//python interface require some access to solver internals,
//so just use the internal crate definitions instead of the API.
use clarabel_solver::core::{
    IPSolver,
    SolverStatus};
use clarabel_solver::implementations::default::*;
use crate::*;



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
// DefaultSolveResult
// ----------------------------------

#[derive(Debug)]
#[pyclass(name = "DefaultSolveResult")]
pub struct PyDefaultSolveResult {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub s: Vec<f64>,
    #[pyo3(get)]
    pub z: Vec<f64>,
    #[pyo3(get)]
    pub obj_val: f64,
    #[pyo3(get)]
    pub status: PySolverStatus,
}

impl PyDefaultSolveResult {
    pub(crate) fn new_from_internal(result: &DefaultSolveResult<f64>) -> Self {
        let x = result.x.clone();
        let s = result.s.clone();
        let z = result.z.clone();
        let obj_val = result.obj_val;
        let status = PySolverStatus::new_from_internal(&result.status);
        Self {
            x,
            s,
            z,
            obj_val,
            status,
        }
    }
}

#[pymethods]
impl PyDefaultSolveResult {
    pub fn __repr__(&self) -> String {
        "Clarabel solution object".to_string()
    }
}

// ----------------------------------
// Solver Status
// ----------------------------------

#[derive(Debug, Clone)]
#[pyclass(name = "SolverStatus")]
pub enum PySolverStatus {
    Unsolved,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterations,
    MaxTime,
}

impl PySolverStatus {
    pub(crate) fn new_from_internal(status: &SolverStatus) -> Self {
        match status {
            SolverStatus::Unsolved => PySolverStatus::Unsolved,
            SolverStatus::Solved => PySolverStatus::Solved,
            SolverStatus::PrimalInfeasible => PySolverStatus::PrimalInfeasible,
            SolverStatus::DualInfeasible => PySolverStatus::DualInfeasible,
            SolverStatus::MaxIterations => PySolverStatus::MaxIterations,
            SolverStatus::MaxTime => PySolverStatus::MaxTime,
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
            PySolverStatus::MaxIterations => "MaxIterations",
            PySolverStatus::MaxTime => "MaxTime",
        }
        .to_string()
    }
}

// ----------------------------------
// Solver Settings
// ----------------------------------

#[derive(Debug,Clone)]
#[pyclass(name = "DefaultSettings")]
pub struct PyDefaultSettings {

    #[pyo3(get, set)]
    pub max_iter: u32,
    #[pyo3(get, set)]
    pub time_limit: f64,
    #[pyo3(get, set)]
    pub verbose: bool,
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
    pub max_step_fraction: f64,

    // data equilibration
    #[pyo3(get, set)]
    pub equilibrate_enable: bool,
    #[pyo3(get, set)]
    pub equilibrate_max_iter: u32,
    #[pyo3(get, set)]
    pub equilibrate_min_scaling: f64,
    #[pyo3(get, set)]
    pub equilibrate_max_scaling: f64,

    // KKT settings incomplete
    #[pyo3(get, set)]
    pub direct_kkt_solver: bool,

    // static regularization parameters
    #[pyo3(get, set)]
    pub static_regularization_enable: bool,
    #[pyo3(get, set)]
    pub static_regularization_eps: f64,

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
}

#[pymethods]
impl PyDefaultSettings {
    #[new]
    pub fn new() -> Self {
        PyDefaultSettings::new_from_internal(&DefaultSettings::<f64>::default())
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    pub fn py_default() -> Self{
        PyDefaultSettings::default()
    }

    pub fn __repr__(&self) -> String {
        let mut s = String::new();
        write!(s, "{:#?}",self.to_internal()).unwrap();
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
            max_iter:                           set.max_iter, 
            time_limit:                         set.time_limit.as_secs_f64(), 
            verbose:                            set.verbose, 
            tol_gap_abs:                        set.tol_gap_abs, 
            tol_gap_rel:                        set.tol_gap_rel, 
            tol_feas:                           set.tol_feas, 
            tol_infeas_abs:                     set.tol_infeas_abs, 
            tol_infeas_rel:                     set.tol_infeas_rel, 
            max_step_fraction:                  set.max_step_fraction, 
            equilibrate_enable:                 set.equilibrate_enable, 
            equilibrate_max_iter:               set.equilibrate_max_iter, 
            equilibrate_min_scaling:            set.equilibrate_min_scaling, 
            equilibrate_max_scaling:            set.equilibrate_max_scaling, 
            direct_kkt_solver:                  set.direct_kkt_solver, 
            static_regularization_enable:       set.static_regularization_enable, 
            static_regularization_eps:          set.static_regularization_eps, 
            dynamic_regularization_enable:      set.dynamic_regularization_enable, 
            dynamic_regularization_eps:         set.dynamic_regularization_eps, 
            dynamic_regularization_delta:       set.dynamic_regularization_delta, 
            iterative_refinement_enable:        set.iterative_refinement_enable, 
            iterative_refinement_reltol:        set.iterative_refinement_reltol, 
            iterative_refinement_abstol:        set.iterative_refinement_abstol, 
            iterative_refinement_max_iter:      set.iterative_refinement_max_iter, 
            iterative_refinement_stop_ratio:    set.iterative_refinement_stop_ratio
        }
    }

    pub(crate) fn to_internal(&self) -> DefaultSettings<f64> {

        DefaultSettings::<f64>{ 
            max_iter:                           self.max_iter, 
            time_limit:                         std::time::Duration::from_secs_f64(self.time_limit), 
            verbose:                            self.verbose, 
            tol_gap_abs:                        self.tol_gap_abs, 
            tol_gap_rel:                        self.tol_gap_rel, 
            tol_feas:                           self.tol_feas, 
            tol_infeas_abs:                     self.tol_infeas_abs, 
            tol_infeas_rel:                     self.tol_infeas_rel, 
            max_step_fraction:                  self.max_step_fraction, 
            equilibrate_enable:                 self.equilibrate_enable, 
            equilibrate_max_iter:               self.equilibrate_max_iter, 
            equilibrate_min_scaling:            self.equilibrate_min_scaling, 
            equilibrate_max_scaling:            self.equilibrate_max_scaling, 
            direct_kkt_solver:                  self.direct_kkt_solver, 
            static_regularization_enable:       self.static_regularization_enable, 
            static_regularization_eps:          self.static_regularization_eps, 
            dynamic_regularization_enable:      self.dynamic_regularization_enable, 
            dynamic_regularization_eps:         self.dynamic_regularization_eps, 
            dynamic_regularization_delta:       self.dynamic_regularization_delta, 
            iterative_refinement_enable:        self.iterative_refinement_enable, 
            iterative_refinement_reltol:        self.iterative_refinement_reltol, 
            iterative_refinement_abstol:        self.iterative_refinement_abstol, 
            iterative_refinement_max_iter:      self.iterative_refinement_max_iter, 
            iterative_refinement_stop_ratio:    self.iterative_refinement_stop_ratio
        }
    }
}



// ----------------------------------
// Solver 
// ----------------------------------

//PJG: not clear if this really needs to be 
//unsendable (i.e. not sendable between threads)
//marked unsendable for now since compilation fails 
//with complaints that some internal substructs need 
//to support the Send trait
#[pyclass(unsendable,name = "DefaultSolver")]
pub struct PyDefaultSolver{
    inner: DefaultSolver<f64>,
}

#[pymethods]
impl PyDefaultSolver {

    #[new]
    fn new(P: PyCscMatrix, 
        q: Vec<f64>, 
        A: PyCscMatrix, 
        b: Vec<f64>, 
        cones: Vec<PySupportedCones>,
        settings: PyDefaultSettings) -> Self {

        let cones = _py_to_native_cones(cones);
        let settings = settings.to_internal();
        let solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

        Self{inner: solver}
    }

    fn solve(&mut self) -> PyDefaultSolveResult {
        
        self.inner.solve();
        PyDefaultSolveResult::new_from_internal(&self.inner.result)
    }


}