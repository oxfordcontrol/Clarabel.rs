#![allow(clippy::missing_transmute_annotations)]

use pyo3::prelude::*;
use pyo3::types::PyCapsule;

// public blas interface
mod blas_wrappers;
pub use blas_wrappers::*;

// public lapack interface
mod lapack_wrappers;
pub use lapack_wrappers::*;

mod blas_loader;
mod blas_types;
mod lapack_loader;
mod lapack_types;

// a function to force instantiation of the blas/lapack wrappers
// stored in lazy_statics.   This function can be called during
// initialization of the python module to ensure that lazy_statics
// are already realised before making an FFI call to blas/lapack.
pub fn force_load() {
    blas_wrappers::force_load();
    lapack_wrappers::force_load();
}

// utilities for scipy blas/lapack import
macro_rules! get_ptr {
    ($api: ident, $str: tt) => {
        std::mem::transmute(get_capsule_void_ptr(&$api, $str)?)
    };
}
pub(crate) use get_ptr;

fn get_capsule_void_ptr(pyx_capi: &Bound<PyAny>, name: &str) -> PyResult<*mut libc::c_void> {
    let binding = pyx_capi.get_item(name)?;
    let capsule: &Bound<PyCapsule> = binding.downcast()?;
    Ok(capsule.pointer())
}

fn get_pyx_capi<'a>(py: Python<'a>, pymodule: &str) -> PyResult<Bound<'a, PyAny>> {
    let lib = PyModule::import(py, pymodule)?;
    let api = lib.getattr("__pyx_capi__")?;
    Ok(api)
}
