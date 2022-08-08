#![allow(non_snake_case)]

use super::types::*;
use crate::solver::{
    core::{
        cones::{SupportedCone, SupportedCone::*},
        IPSolver,
    },
    implementations::default::*,
};
use num_traits::FromPrimitive;
use serde_json::*;
use std::{ffi::CStr, os::raw::c_void};

// functions for converting solver to / from c void pointers

fn to_ptr(solver: Box<DefaultSolver<f64>>) -> *mut c_void {
    Box::into_raw(solver) as *mut c_void
}

fn from_ptr(ptr: *mut c_void) -> Box<DefaultSolver<f64>> {
    unsafe { Box::from_raw(ptr as *mut DefaultSolver<f64>) }
}

// function for receiving cone specifications in rust
// in a flattened form from Julia

fn ccall_arrays_to_cones(
    cones_enums: &VectorJLRS<u8>,
    cones_ints: &VectorJLRS<u64>,
    cones_floats: &VectorJLRS<f64>,
) -> Vec<SupportedCone<f64>> {
    let mut cones: Vec<SupportedCone<f64>> = Vec::new();

    assert_eq!(cones_enums.len(), cones_ints.len());
    assert_eq!(cones_enums.len(), cones_floats.len());

    // convert to rust vector types from raw pointers
    let cones_enums = Vec::from(cones_enums);
    let cones_ints = Vec::from(cones_ints);
    let _cones_float = Vec::from(cones_floats);

    for i in 0..cones_enums.len() {
        let cone = match FromPrimitive::from_u8(cones_enums[i]) {
            Some(ConeEnumJLRS::ZeroConeT) => ZeroConeT(cones_ints[i] as usize),
            Some(ConeEnumJLRS::NonnegativeConeT) => NonnegativeConeT(cones_ints[i] as usize),
            Some(ConeEnumJLRS::SecondOrderConeT) => SecondOrderConeT(cones_ints[i] as usize),
            None => panic!("Received unrecognized cone type"),
        };
        cones.push(cone)
    }
    cones
}

#[no_mangle]
pub(crate) extern "C" fn solver_new_jlrs(
    P: &CscMatrixJLRS,
    q: &VectorJLRS<f64>,
    A: &CscMatrixJLRS,
    b: &VectorJLRS<f64>,
    cones_enums: &VectorJLRS<u8>,
    cones_ints: &VectorJLRS<u64>,
    cones_floats: &VectorJLRS<f64>,
    json_settings: *const std::os::raw::c_char,
) -> *mut c_void {
    let P = P.to_CscMatrix();
    let A = A.to_CscMatrix();
    let q = Vec::from(q);
    let b = Vec::from(b);

    let cones = ccall_arrays_to_cones(cones_enums, cones_ints, cones_floats);

    let settings = settings_from_json(json_settings);

    let solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    to_ptr(Box::new(solver))
}

#[no_mangle]
pub(crate) extern "C" fn solver_solve_jlrs(ptr: *mut c_void) -> SolutionJLRS {
    let mut solver = from_ptr(ptr);
    solver.solve();

    let out = SolutionJLRS::from(&solver.solution);

    // don't drop, since the memory is owned by
    // Julia and we might want to solve again
    std::mem::forget(solver);

    out
}

#[no_mangle]
pub(crate) extern "C" fn solver_get_info_jlrs(ptr: *mut c_void) -> DefaultInfo<f64> {
    println!("Entering solver_get_info_jlrs");

    let solver = from_ptr(ptr);

    let info = solver.info.clone();

    // don't drop, since the memory is owned by
    // Julia and we might want to solve again
    std::mem::forget(solver);

    info
}

// safely drop a solver object through its pointer.
// called by the Julia side finalizer when a solver
// is out of scope
#[no_mangle]
pub(crate) extern "C" fn solver_drop_jlrs(ptr: *mut c_void) {
    drop(from_ptr(ptr));
}

pub(crate) fn settings_from_json(json: *const std::os::raw::c_char) -> DefaultSettings<f64> {
    // convert julia side json to Rust str
    let json = unsafe {
        let slice = CStr::from_ptr(json);
        slice.to_str().unwrap()
    };

    let mut settings: DefaultSettings<f64> = from_str(json).unwrap();

    // Julia serializes Inf => None, so Julia side
    // converts to f64::MAX before serialization
    if settings.time_limit > f64::MAX * 0.9999 {
        settings.time_limit = f64::INFINITY;
    }
    settings
}
