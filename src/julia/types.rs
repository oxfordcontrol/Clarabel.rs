#![allow(non_snake_case)]

use crate::algebra::CscMatrix;
use crate::solver::core::kktsolvers::LinearSolverInfo;
use crate::solver::implementations::default::*;
use num_derive::FromPrimitive;
use std::slice;

// The types defined here are for exchanging data
// between Rust and Julia.

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct VectorJLRS<T> {
    pub p: *const T,
    pub len: libc::size_t,
}

#[derive(Debug, Clone)]
#[repr(C)]
pub(crate) struct ConeDataJLRS {
    pub tag: u8,
    pub int: usize,
    pub float: f64,
    pub vec: VectorJLRS<f64>,
}

impl<T> VectorJLRS<T>
where
    T: std::clone::Clone + std::fmt::Debug,
{
    pub(crate) fn to_slice(&self) -> &[T] {
        assert!(!self.p.is_null());
        unsafe { slice::from_raw_parts(self.p, self.len) }
    }

    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

// The following conversions are used to convert from
// Julia problem data into Rust types.

impl From<&VectorJLRS<i64>> for Vec<usize> {
    fn from(v: &VectorJLRS<i64>) -> Self {
        v.to_slice().iter().map(|&e| e as usize).collect()
    }
}

impl<T> From<&VectorJLRS<T>> for Vec<T>
where
    T: std::clone::Clone + std::fmt::Debug,
{
    fn from(v: &VectorJLRS<T>) -> Self {
        let sl = if v.len > 0 { v.to_slice() } else { &[] };
        sl.to_vec()
    }
}

// The following conversions are used to convert from Rust data
// back to Julia.   Here we only need vectors of floats.

impl<T> From<&Vec<T>> for VectorJLRS<T>
where
    T: std::clone::Clone + std::fmt::Debug,
{
    fn from(v: &Vec<T>) -> Self {
        let mut v = v.clone();
        let p = v.as_mut_ptr();
        let len = v.len();
        std::mem::forget(v);
        VectorJLRS::<T> { p, len }
    }
}

#[derive(Debug)]
#[repr(C)]
pub(crate) struct CscMatrixJLRS {
    pub m: usize,
    pub n: usize,
    pub colptr: VectorJLRS<i64>,
    pub rowval: VectorJLRS<i64>,
    pub nzval: VectorJLRS<f64>,
}

impl CscMatrixJLRS {
    pub(crate) fn to_CscMatrix(&self) -> CscMatrix {
        let mut colptr = Vec::<usize>::from(&self.colptr);
        let mut rowval = Vec::<usize>::from(&self.rowval);
        let nzval = Vec::<f64>::from(&self.nzval);

        //need to back shift all of the indices since
        //Julia stores matrices as 1-indexed
        for v in rowval.iter_mut() {
            *v -= 1
        }
        for v in colptr.iter_mut() {
            *v -= 1
        }
        CscMatrix::new(self.m, self.n, colptr, rowval, nzval)
    }
}

#[repr(C)]
#[derive(Debug)]
pub(crate) struct SolutionJLRS {
    pub x: VectorJLRS<f64>,
    pub z: VectorJLRS<f64>,
    pub s: VectorJLRS<f64>,
    pub status: u32, //0 indexed enum in RS/JL
    pub obj_val: f64,
    pub obj_val_dual: f64,
    pub solve_time: f64,
    pub iterations: u32,
    pub r_prim: f64,
    pub r_dual: f64,
}

impl From<&DefaultSolution<f64>> for SolutionJLRS {
    fn from(sol: &DefaultSolution<f64>) -> Self {
        SolutionJLRS {
            x: VectorJLRS::<f64>::from(&sol.x),
            z: VectorJLRS::<f64>::from(&sol.z),
            s: VectorJLRS::<f64>::from(&sol.s),
            status: sol.status as u32,
            obj_val: sol.obj_val,
            obj_val_dual: sol.obj_val_dual,
            solve_time: sol.solve_time,
            iterations: sol.iterations,
            r_prim: sol.r_prim,
            r_dual: sol.r_dual,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct LinearSolverInfoJLRS {
    pub name: VectorJLRS<u8>,
    pub threads: usize,
    pub direct: bool,
    pub nnzA: usize,
    pub nnzL: usize,
}

impl From<&LinearSolverInfo> for LinearSolverInfoJLRS {
    fn from(sol: &LinearSolverInfo) -> Self {
        LinearSolverInfoJLRS {
            name: VectorJLRS::<u8>::from(&sol.name.as_bytes().to_vec()),
            threads: sol.threads,
            direct: sol.direct,
            nnzA: sol.nnzA,
            nnzL: sol.nnzL,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub(crate) struct InfoJLRS {
    pub mu: f64,
    pub sigma: f64,
    pub step_length: f64,
    pub iterations: u32,
    pub cost_primal: f64,
    pub cost_dual: f64,
    pub res_primal: f64,
    pub res_dual: f64,
    pub res_primal_inf: f64,
    pub res_dual_inf: f64,
    pub gap_abs: f64,
    pub gap_rel: f64,
    pub ktratio: f64,
    pub prev_cost_primal: f64,
    pub prev_cost_dual: f64,
    pub prev_res_primal: f64,
    pub prev_res_dual: f64,
    pub prev_gap_abs: f64,
    pub prev_gap_rel: f64,
    pub solve_time: f64,
    pub status: u32, //0 indexed enum in RS/JL
    //NB : print stream left out because it is not FFI safe
    pub linsolver: LinearSolverInfoJLRS,
}

impl From<&DefaultInfo<f64>> for InfoJLRS {
    fn from(sol: &DefaultInfo<f64>) -> Self {
        InfoJLRS {
            mu: sol.mu,
            sigma: sol.sigma,
            step_length: sol.step_length,
            iterations: sol.iterations,
            cost_primal: sol.cost_primal,
            cost_dual: sol.cost_dual,
            res_primal: sol.res_primal,
            res_dual: sol.res_dual,
            res_primal_inf: sol.res_primal_inf,
            res_dual_inf: sol.res_dual_inf,
            gap_abs: sol.gap_abs,
            gap_rel: sol.gap_rel,
            ktratio: sol.ktratio,
            prev_cost_primal: sol.prev_cost_primal,
            prev_cost_dual: sol.prev_cost_dual,
            prev_res_primal: sol.prev_res_primal,
            prev_res_dual: sol.prev_res_dual,
            prev_gap_abs: sol.prev_gap_abs,
            prev_gap_rel: sol.prev_gap_rel,
            solve_time: sol.solve_time,
            status: sol.status as u32,
            linsolver: LinearSolverInfoJLRS::from(&sol.linsolver),
        }
    }
}

#[repr(u8)]
#[derive(FromPrimitive)]
pub(crate) enum ConeEnumJLRS {
    ZeroConeT = 0,
    NonnegativeConeT = 1,
    SecondOrderConeT = 2,
    ExponentialConeT = 3,
    PowerConeT = 4,
    GenPowerConeT = 5,
    PSDTriangleConeT = 6,
}
