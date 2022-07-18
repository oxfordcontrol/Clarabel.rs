#![allow(unused)]
#![allow(non_snake_case)]
use std::slice;

//julia interface require some access to solver internals, 
//so just use the internal crate definitions instead of the API.
use clarabel_algebra as algebra;
use clarabel_solver  as solver;
use algebra::CscMatrix;
use solver::implementations::default::*;
use solver::core::{cones::SupportedCones::*, 
                   IPSolver};

#[repr(C)]
#[derive(Debug)]
pub struct VectorJl<T> {
    p: *const T,
    len: libc::size_t,
}

impl<T> VectorJl<T> 
where 
    T: std::clone::Clone + std::fmt::Debug
{
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
) -> f64 

{
    let P = P.to_CscMatrix();
    let A = A.to_CscMatrix();
    let q = q.to_slice().to_vec();
    let b = b.to_slice().to_vec();

    let cones = [NonnegativeConeT(b.len())];

    println!("P = {:?}",P);
    println!("A = {:?}",A);
    println!("q = {:?}",q);
    println!("b = {:?}",b);

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(20)
        .verbose(true)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();

    3.
}
