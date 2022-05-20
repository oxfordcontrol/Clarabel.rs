#![allow(unused)]
#![allow(non_snake_case)]
use std::slice;

use clarabel_internal as clarabel;

// PJG: probably need better re-export here so only clarabel::* suffices.
// PJG: this "solver::solver" stuff sucks here.
use clarabel::*;
use clarabel::solver::SupportedCones::*;
use clarabel::solver::settings::*;
use clarabel::solver::solver::IPSolver;
use clarabel::solver::default::*;
use clarabel::qdldl::*;
use algebra::*;

#[repr(C)]
#[derive(Debug)]
pub struct VectorJl<T> {
    p: *const T, 
    len: libc::size_t 
}

impl<T: std::clone::Clone + std::fmt::Debug> VectorJl<T> {
    fn to_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.p, self.len as usize)
        }
    }

    fn to_vec(&self) -> Vec<T> {

        assert!(!self.p.is_null());
        let sl;
        if(self.len > 0){
            unsafe {
            sl = slice::from_raw_parts(self.p, self.len);
            } 
        }
        else {
             sl = &[];
        }
        sl.to_vec()
    }
}


#[derive(Debug)]
#[repr(C)]
pub struct CscMatrixJl{
    m: usize,
    n: usize,
    pub colptr: VectorJl<i64>,
    pub rowval: VectorJl<i64>,
    pub nzval: VectorJl<f64>,
}

impl CscMatrixJl {
    fn to_CscMatrix(&self) -> CscMatrix {

        //println!("Getting colptr");
        let colptr = self.colptr.
                     to_slice().
                     iter().
                     map(|&e| e as usize).
                     collect();
        //println!("Getting rowval");
        let rowval = self.rowval.to_slice().
                     iter().
                     map(|&e| e as usize).
                     collect();
        //println!("Getting nzval");
        let nzval  = self.nzval.to_vec();

        let mut m = CscMatrix{
            m: self.m,
            n: self.n,
            colptr,
            rowval,
            nzval,
        };

        //need to back shift all of the indices since 
        //Julia stores matrices as 1-indexed 
        for v in m.rowval.iter_mut() {*v -= 1}
        for v in m.colptr.iter_mut() {*v -= 1}
        
        m //output
    }
}


#[no_mangle]
pub extern fn catch_matrix(s: &CscMatrixJl) -> i32 {

    println!("\n\nCscMatrixJl is {:?}\n\n", s);

    let m = s.to_CscMatrix();

    println!("\n\nCscMatrix is {:?}\n\n", m);
    4
}

#[no_mangle]
pub extern fn solve(
    P: &CscMatrixJl, 
    q: &VectorJl<f64>, 
    A: &CscMatrixJl, 
    b: &VectorJl<f64>) -> f64 
{
    let P = P.to_CscMatrix();
    let A = A.to_CscMatrix();
    let q = q.to_slice().to_vec();
    let b = b.to_slice().to_vec();

    let cone_types = [NonnegativeConeT];

    let cone_dims  = [b.len()];

    let settings = SettingsBuilder::default()
            .equilibrate_enable(true)
            .max_iter(0)
            .verbose(true)
            .build().unwrap();

    //PJG: no borrow on settings sucks here
    let mut solver = DefaultSolver::
        new(&P,&q,&A,&b,&cone_types,&cone_dims, settings);

    solver.solve();

    3.
}

#[no_mangle]
pub extern fn amd(
    K: &CscMatrixJl) -> f64 
{
    println!("Entering AMD rust side interface");
    let K = K.to_CscMatrix();
    println!("created K");

    let t = std::time::Instant::now();
    let (p,ip) = _get_amd_ordering(&K);
    println!("AMD Time: {:?}",t.elapsed());
    3.
}

#[no_mangle]
pub extern fn printmat(
    K: &CscMatrixJl)
{
    println!("Entering AMD rust side interface");
    let K = K.to_CscMatrix();
    println!("created K");
}

#[no_mangle]
pub extern fn printvec_u64(
    v: &VectorJl<u64>)
{
    println!("Entering AMD rust side interface");
    let v = v.to_vec();
    println!("created K");
}

#[no_mangle]
pub extern fn printvec_f64(
    v: &VectorJl<f64>)
{
    println!("Entering AMD rust side interface");
    let v = v.to_vec();
    println!("created K");
}


#[no_mangle]
pub extern fn symdot(
    K: &CscMatrixJl, x: &VectorJl<f64>, y: &VectorJl<f64>) -> f64 
{
    let K = K.to_CscMatrix();
    let x = x.to_vec();
    let y = y.to_vec();

    let t = std::time::Instant::now();
    let val = K.symdot(&x,&y);
    let stop = t.elapsed();
    println!("SYM  prod time: {:?}.  val = {:?}",stop, val);

    let t = std::time::Instant::now();
    let val = _fast_quad_form(&K,&x,&y);
    let stop = t.elapsed();
    println!("FAST prod time: {:?}. val = {:?} ",stop, val);

    val
}

#[no_mangle]
pub extern fn qdldl(K: &CscMatrixJl) {

    let K = K.to_CscMatrix();

    //construct the LDL solver settings
    let opts = QDLDLSettingsBuilder::default()
    .logical(false) //allocate memory only on init
    .build()
    .unwrap();

    let t = std::time::Instant::now();
    let factors = QDLDLFactorisation::<f64>::new(&K, Some(opts));
    let stop = t.elapsed();
    println!("factor time: {:?}",stop);
}


#[no_mangle]
pub extern fn mydot(x: &VectorJl<f64>, y: &VectorJl<f64>) -> f64 
{
    let x = x.to_vec();
    let y = y.to_vec();

    let t = std::time::Instant::now();
    let val = _fastdot8(&x,&y);
    let stop = t.elapsed();
    println!("unrolled 8 dot time: {:?}.  Value = {:?}",stop, val);

    let t = std::time::Instant::now();
    let val = x.dot(&y);
    let stop = t.elapsed();
    println!("regular  dot time: {:?}.  Value = {:?}",stop, val);

    let t = std::time::Instant::now();
    let val = _flatdot(&x,&y);
    let stop = t.elapsed();
    println!("flat dot time: {:?}.  Value = {:?}",stop, val);

    val
}

#[inline(never)]
fn _flatdot(x: &[f64], y:&[f64]) -> f64{ 

    let mut out = 0.0;

    let len = std::cmp::min(x.len(),y.len());

    let mut x = &x[..len];
    let mut y = &y[..len];


    for (&xi,&yi) in x.iter().zip(y){
        out += xi*yi;
    }
    out
}

#[inline(never)]
fn _fastdot8(x: &[f64], y:&[f64]) -> f64{ 

    let mut out = 0.0;

    let len = std::cmp::min(x.len(),y.len());

    let mut x = &x[..len];
    let mut y = &y[..len];

    while x.len() >= 8 {
        out += x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3] + 
              x[4]*y[4] + x[5]*y[5] + x[6]*y[6] + x[7]*y[7] ; 

        x = &x[8..];
        y = &y[8..];
    }

    for (&xi,&yi) in x.iter().zip(y){
        out += xi*yi;
    }
    out

}

#[allow(non_snake_case)]
#[allow(clippy::comparison_chain)]
fn _fast_quad_form<T: FloatT>(M: &CscMatrix<T>, y: &[T], x: &[T]) -> T{

    assert_eq!(M.n, M.m);
    assert_eq!(x.len(), M.n);
    assert_eq!(y.len(), M.n);
    assert!(M.colptr.len() == M.n+1);
    assert!(M.nzval.len() == M.rowval.len());

    if M.n == 0 {
        return T::zero()
    }

    let mut out = T::zero();

    for col in 0..M.n {   //column number

        let mut tmp1 = T::zero();
        let mut tmp2 = T::zero();

        //start / stop indices for the current column
        let first = M.colptr[col];
        let last  = M.colptr[col+1];

        let values   = &M.nzval[first..last];
        let rows     = &M.rowval[first..last];
        let iter     = values.iter().zip(rows.iter());

        for (&Mv,&row) in iter
         {
            if row < col {
                //triu terms only
                tmp1 += Mv*x[row];
                tmp2 += Mv*y[row];
            }
            else if row == col {
                out += Mv*x[col]*y[col];
            }
            else{
                panic!("Input matrix should be triu form.");
            }   
        }
        out += tmp1*y[col] + tmp2*x[col]
    }
    out
}




fn _fast_sparse_dot8(x: &[f64], xidx: &[usize], ys:&[f64]) -> f64{ 

    let mut out = 0.0;

    let len = std::cmp::min(ys.len(),xidx.len());

    let mut xidx = &xidx[..len];
    let mut ys = &ys[..len];

    while xidx.len() >= 8 {
        out += x[xidx[0]]*ys[0] + 
               x[xidx[1]]*ys[1] + 
               x[xidx[2]]*ys[2] + 
               x[xidx[3]]*ys[3] + 
               x[xidx[4]]*ys[4] + 
               x[xidx[5]]*ys[5] + 
               x[xidx[6]]*ys[6] + 
               x[xidx[7]]*ys[7];

        xidx = &xidx[8..];
        ys = &ys[8..];
    }

    for (&i,&ysi) in xidx.iter().zip(ys) {
        out += x[i]*ysi;
    }
    out
}