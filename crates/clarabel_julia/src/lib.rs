#![allow(unused)]
#![allow(non_snake_case)]
use std::slice;

use clarabel_internal as clarabel;

// PJG: probably need better re-export here so only clarabel::* suffices.
// PJG: this "solver::solver" stuff sucks here.
//Not sure I understand why "algebra" even works here
use algebra::*;
use clarabel::qdldl::*;
use clarabel::solver::default::*;
use clarabel::solver::settings::*;
use clarabel::solver::solver::IPSolver;
use clarabel::solver::SupportedCones::*;
use clarabel::*;

#[repr(C)]
#[derive(Debug)]
pub struct VectorJl<T> {
    p: *const T,
    len: libc::size_t,
}

impl<T: std::clone::Clone + std::fmt::Debug> VectorJl<T> {
    fn to_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.p, self.len as usize) }
    }

    fn to_vec(&self) -> Vec<T> {
        assert!(!self.p.is_null());
        let sl;
        if (self.len > 0) {
            unsafe {
                sl = slice::from_raw_parts(self.p, self.len);
            }
        } else {
            sl = &[];
        }
        sl.to_vec()
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct CscMatrixJl {
    m: usize,
    n: usize,
    pub colptr: VectorJl<i64>,
    pub rowval: VectorJl<i64>,
    pub nzval: VectorJl<f64>,
}

impl CscMatrixJl {
    fn to_CscMatrix(&self) -> CscMatrix {
        //println!("Getting colptr");
        let colptr = self.colptr.to_slice().iter().map(|&e| e as usize).collect();
        //println!("Getting rowval");
        let rowval = self.rowval.to_slice().iter().map(|&e| e as usize).collect();
        //println!("Getting nzval");
        let nzval = self.nzval.to_vec();

        let mut m = CscMatrix {
            m: self.m,
            n: self.n,
            colptr,
            rowval,
            nzval,
        };

        //need to back shift all of the indices since
        //Julia stores matrices as 1-indexed
        for v in m.rowval.iter_mut() {
            *v -= 1
        }
        for v in m.colptr.iter_mut() {
            *v -= 1
        }

        m //output
    }
}


#[no_mangle]
pub extern "C" fn solve(
    P: &CscMatrixJl,
    q: &VectorJl<f64>,
    A: &CscMatrixJl,
    b: &VectorJl<f64>,
) -> f64 {
    let P = P.to_CscMatrix();
    let A = A.to_CscMatrix();
    let q = q.to_slice().to_vec();
    let b = b.to_slice().to_vec();

    let cone_types = [NonnegativeConeT(b.len())];

    let settings = SettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(0)
        .verbose(true)
        .build()
        .unwrap();

    //PJG: no borrow on settings sucks here
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cone_types, settings);

    solver.solve();

    3.
}
