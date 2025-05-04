#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::upper_case_acronyms)]
#![allow(non_camel_case_types)]

use crate::solver::{LinearSolverInfo, SolverStatus};

/// FFI interface type for [`LinearSolverInfo`](crate::solver::LinearSolverInfo)
#[allow(missing_docs)]
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LinearSolverInfoFFI {
    pub name: DirectSolveMethodsFFI,
    pub threads: u32,
    pub direct: bool,
    pub nnzA: u32,
    pub nnzL: u32,
}

impl From<LinearSolverInfo> for LinearSolverInfoFFI {
    fn from(linsolver: LinearSolverInfo) -> Self {
        Self {
            name: linsolver.name.into(),
            threads: linsolver.threads as u32,
            direct: linsolver.direct,
            nnzA: linsolver.nnzA as u32,
            nnzL: linsolver.nnzL as u32,
        }
    }
}

// No From<LinearSolverInfoFFI> because solver information
// flows only one way

/// FFI interface for direct linear solver methods
#[allow(missing_docs)]
#[repr(C)]
#[derive(Debug, Clone)]
pub enum DirectSolveMethodsFFI {
    AUTO = 0,
    QDLDL = 1,
    #[cfg(feature = "faer-sparse")]
    FAER = 2,
    #[cfg(feature = "pardiso-mkl")]
    MKL = 3,
    #[cfg(feature = "pardiso-panua")]
    PANUA = 4,
}

impl From<DirectSolveMethodsFFI> for String {
    fn from(value: DirectSolveMethodsFFI) -> Self {
        match value {
            DirectSolveMethodsFFI::AUTO => String::from("auto"),
            DirectSolveMethodsFFI::QDLDL => String::from("qdldl"),
            #[cfg(feature = "faer-sparse")]
            DirectSolveMethodsFFI::FAER => String::from("faer"),
            #[cfg(feature = "pardiso-mkl")]
            DirectSolveMethodsFFI::MKL => String::from("mkl"),
            #[cfg(feature = "pardiso-panua")]
            DirectSolveMethodsFFI::PANUA => String::from("panua"),
        }
    }
}

impl From<String> for DirectSolveMethodsFFI {
    fn from(value: String) -> Self {
        match value.as_str() {
            "auto" => DirectSolveMethodsFFI::AUTO,
            "qdldl" => DirectSolveMethodsFFI::QDLDL,
            #[cfg(feature = "faer-sparse")]
            "faer" => DirectSolveMethodsFFI::FAER,
            #[cfg(feature = "pardiso-mkl")]
            "mkl" => DirectSolveMethodsFFI::MKL,
            #[cfg(feature = "pardiso-panua")]
            "panua" => DirectSolveMethodsFFI::PANUA,
            _ => DirectSolveMethodsFFI::AUTO,
        }
    }
}

/// FFI interface type for clique merging methods
#[allow(missing_docs)]
#[cfg(feature = "sdp")]
#[repr(C)]
#[derive(Debug, Clone)]
pub enum CliqueMergeMethodsFFI {
    CLIQUE_GRAPH,
    PARENT_CHILD,
    NONE,
}

#[cfg(feature = "sdp")]
impl From<String> for CliqueMergeMethodsFFI {
    fn from(value: String) -> Self {
        match value.as_str() {
            "clique_graph" => CliqueMergeMethodsFFI::CLIQUE_GRAPH,
            "parent_child" => CliqueMergeMethodsFFI::PARENT_CHILD,
            "none" => CliqueMergeMethodsFFI::NONE,
            _ => CliqueMergeMethodsFFI::NONE,
        }
    }
}

#[cfg(feature = "sdp")]
impl From<CliqueMergeMethodsFFI> for String {
    fn from(value: CliqueMergeMethodsFFI) -> Self {
        match value {
            CliqueMergeMethodsFFI::CLIQUE_GRAPH => String::from("clique_graph"),
            CliqueMergeMethodsFFI::PARENT_CHILD => String::from("parent_child"),
            CliqueMergeMethodsFFI::NONE => String::from("none"),
        }
    }
}

#[allow(missing_docs)]
/// FFI interface for [`SolverStatus`](crate::solver::SolverStatus)
pub type SolverStatusFFI = SolverStatus;

#[test]

fn test_enum_ffis() {
    let obj = LinearSolverInfo {
        threads: 4,
        ..Default::default()
    };
    let obj_ffi = LinearSolverInfoFFI::from(obj.clone());
    assert_eq!(obj.threads, obj_ffi.threads as usize);

    let obj = "qdldl".to_string();
    let obj_ffi = DirectSolveMethodsFFI::from(obj.clone());
    let obj_ffi = String::from(obj_ffi);
    assert_eq!(obj, obj_ffi);
}

#[cfg(feature = "sdp")]
fn test_enum_ffis_sdps() {
    let obj = "clique_graph".to_string();
    let obj_ffi = CliqueMergeMethodsFFI::from(obj.clone());
    let obj_ffi = String::from(obj_ffi);
    assert_eq!(obj, obj_ffi);
}
