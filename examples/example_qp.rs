#![allow(non_snake_case)]

//PJG: Don't understand how to make the import just "algebra" here.
//PJG: Somehow need to bundle all default solver types and re-export
//at a common module level

use clarabel::algebra;
use clarabel::solver;
use crate::default::DefaultSolver;
use clarabel::solver::SupportedCones::*;
use clarabel::solver::solver::IPSolver; //solve fails without this.  Should be easier
use clarabel::solver::settings::SettingsBuilder;  //shouldn't need to explicit import this

fn _problem_data() -> (CscMatrix<f64>,Vec<f64>,CscMatrix<f64>,Vec<f64>)
{

    let P : CscMatrix<f64> =
            CscMatrix{m : 2,
                      n : 2,
                      colptr : vec![0,1,2],
                      rowval : vec![0,1],
                      nzval  : vec![6.,4.]};

    let q = vec![-1.,-4.];

    let A : CscMatrix<f64> =
            CscMatrix{m : 5,
                      n : 2,
                      colptr : vec![0,3,6],
                      rowval : vec![0,1,3,0,2,4],
                      nzval  : vec![1.,1.,-1.,-2.,1.,-1.]};

    let b = vec![0.,1.,1.,1.,1.];

    (P,q,A,b)

}


fn main() {

    let (P,q,A,b) = _problem_data();

    let cone_types = [ZeroConeT, NonnegativeConeT];

    let cone_dims  = [1, 4];

    let settings = SettingsBuilder::default()
            .equilibrate_enable(true)
            .max_iter(50)
            .build().unwrap();

    //PJG: no borrow on settings sucks here
    let mut solver = DefaultSolver::
            new(&P,&q,&A,&b,&cone_types,&cone_dims, settings);

    solver.solve();



}
