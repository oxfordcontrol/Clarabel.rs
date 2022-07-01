#![allow(non_snake_case)]

//PJG: Don't understand how to make the import just "algebra" here.
//PJG: Somehow need to bundle all default solver types and re-export
//at a common module level

//PJG: some includes seem redundant
use clarabel::algebra::*;
use clarabel::solver::implementations::default::*;
use clarabel::solver::SupportedCones::*;
use clarabel::solver::solver::IPSolver; //solve fails without this.  Should be easier
use clarabel::solver::settings::SettingsBuilder;  //shouldn't need to explicit import this

fn _problem_data() -> (CscMatrix<f64>,Vec<f64>,CscMatrix<f64>,Vec<f64>)
{
    let n = 20000;

    let mut P = CscMatrix::<f64>::spalloc(n,n,n);

    for i in 0..n {
        P.colptr[i] = i;
        P.rowval[i] = i;
        P.nzval[i] = 1.;
    }
    P.colptr[n] = n;
   

    let mut A = CscMatrix::<f64>::spalloc(2*n,n,2*n);

    for i in 0..n {
        A.colptr[i] = 2*i;
        A.rowval[2*i] = i;
        A.rowval[2*i+1] = i+n;
        A.nzval[2*i] = 1.;
        A.nzval[2*i+1] = -1.
    }
    A.colptr[n] = 2*n;

    let q = vec![1.;n];
    let b = vec![1.;2*n];

    (P,q,A,b)

}


fn main() {

    let (P,q,A,b) = _problem_data();

    let cone_types = [NonnegativeConeT(b.len())];

    let settings = SettingsBuilder::default()
            .equilibrate_enable(true)
            .max_iter(50)
            .build().unwrap();

    //PJG: no borrow on settings sucks here
    let mut solver = DefaultSolver::
            new(&P,&q,&A,&b,&cone_types,settings);

    solver.solve();



}
