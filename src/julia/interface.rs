#![allow(non_snake_case)]

use super::types::*;
use crate::solver::{
    core::{
        cones::{SupportedConeT, SupportedConeT::*},
        IPSolver,
    },
    implementations::default::*,
    SolverJSONReadWrite,
};
use num_traits::FromPrimitive;
use serde_json::*;
use std::fs::File;
use std::{
    ffi::CStr,
    os::raw::{c_char, c_int, c_void},
};

// functions for converting solver to / from c void pointers

fn to_ptr(solver: Box<DefaultSolver<f64>>) -> *mut c_void {
    Box::into_raw(solver) as *mut c_void
}

fn from_ptr(ptr: *mut c_void) -> Box<DefaultSolver<f64>> {
    unsafe { Box::from_raw(ptr as *mut DefaultSolver<f64>) }
}

// function for sending cone specifications into rust
// from a tagged union type form supplied from Julia

fn ccall_arrays_to_cones(jlcones: &VectorJLRS<ConeDataJLRS>) -> Vec<SupportedConeT<f64>> {
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();

    for jlcone in jlcones.to_slice() {
        let cone = match FromPrimitive::from_u8(jlcone.tag) {
            Some(ConeEnumJLRS::ZeroConeT) => ZeroConeT(jlcone.int),
            Some(ConeEnumJLRS::NonnegativeConeT) => NonnegativeConeT(jlcone.int),
            Some(ConeEnumJLRS::SecondOrderConeT) => SecondOrderConeT(jlcone.int),
            Some(ConeEnumJLRS::ExponentialConeT) => ExponentialConeT(),
            Some(ConeEnumJLRS::PowerConeT) => PowerConeT(jlcone.float),
            Some(ConeEnumJLRS::GenPowerConeT) => {
                let alpha = Vec::<f64>::from(&jlcone.vec);
                GenPowerConeT(alpha, jlcone.int)
            }
            Some(ConeEnumJLRS::PSDTriangleConeT) => PSDTriangleConeT(jlcone.int),
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
    jlcones: &VectorJLRS<ConeDataJLRS>,
    json_settings: *const c_char,
) -> *mut c_void {
    let P = P.to_CscMatrix();
    let A = A.to_CscMatrix();
    let q = Vec::from(q);
    let b = Vec::from(b);

    let cones = ccall_arrays_to_cones(jlcones);
    let settings = settings_from_json(json_settings);

    // manually validate settings from Julia side
    match settings.validate() {
        Ok(_) => (),
        Err(e) => {
            println!("Invalid settings: {}", e);
            return std::ptr::null_mut();
        }
    };

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
    let solver = from_ptr(ptr);

    let info = solver.info.clone();

    // don't drop, since the memory is owned by
    // Julia and we might want to solve again
    std::mem::forget(solver);

    info
}

#[no_mangle]
pub(crate) extern "C" fn solver_print_timers_jlrs(ptr: *mut c_void) {
    let solver = from_ptr(ptr);

    match solver.timers {
        Some(ref timers) => timers.print(),
        None => println!("No timer info available"),
    }

    // don't drop, since the memory is owned by
    // Julia and we might want to solve again
    std::mem::forget(solver);
}

// dump problem data to a file
// returns -1 on failure, 0 on success
#[no_mangle]
pub(crate) extern "C" fn solver_write_to_file_jlrs(
    ptr: *mut c_void,
    filename: *const std::os::raw::c_char,
) -> c_int {
    let slice = unsafe { CStr::from_ptr(filename) };

    let filename = match slice.to_str() {
        Ok(s) => s,
        Err(_) => {
            return -1;
        }
    };

    let mut file = match File::create(filename) {
        Ok(f) => f,
        Err(_) => {
            return -1;
        }
    };

    let solver = from_ptr(ptr);
    let status = solver.write_to_file(&mut file).is_ok();
    let status = if status { 0 } else { -1 } as c_int;

    // don't drop, since the memory is owned by Julia
    std::mem::forget(solver);

    status
}

// read problem data from a file with serialized JSON settings
// returns NULL on failure, pointer to solver on success
#[no_mangle]
pub(crate) extern "C" fn solver_read_from_file_jlrs(
    filename: *const std::os::raw::c_char,
    json_settings: *const std::os::raw::c_char,
) -> *const c_void {
    let slice = unsafe { CStr::from_ptr(filename) };

    let filename = match slice.to_str() {
        Ok(s) => s,
        Err(_) => {
            return std::ptr::null();
        }
    };

    let mut file = match File::open(filename) {
        Ok(f) => f,
        Err(_) => {
            return std::ptr::null();
        }
    };

    // None on the julia size is serialized as "",
    let settings = unsafe {
        if json_settings.is_null() {
            None
        } else if CStr::from_ptr(json_settings).to_bytes().is_empty() {
            None
        } else {
            Some(settings_from_json(json_settings))
        }
    };

    let solver = DefaultSolver::read_from_file(&mut file, settings);

    match solver {
        Ok(solver) => to_ptr(Box::new(solver)),
        Err(_) => std::ptr::null(),
    }
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

#[no_mangle]
pub(crate) extern "C" fn buildinfo_jlrs() {
    crate::buildinfo();
}
