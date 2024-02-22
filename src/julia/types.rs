#![allow(non_snake_case)]

use crate::algebra::CscMatrix;
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
