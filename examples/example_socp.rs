#![allow(non_snake_case)]

//PJG: Don't understand how to make the import just "algebra" here.
//PJG: Somehow need to bundle all default solver types and re-export
//at a common module level

use algebra::*;
use solver::*;
use crate::default::DefaultSolver;
use clarabel::solver::SupportedCones::*;
use clarabel::solver::IPSolver; //solve fails without this.  Should be easier
use crate::settings::SettingsBuilder;  //shouldn't need to explicit import this

fn _problem_data() -> (CscMatrix<f64>,Vec<f64>,CscMatrix<f64>,Vec<f64>)
{

    let P : CscMatrix<f64> =
    CscMatrix{m : 2,
        n : 2,
        colptr : vec![0,0,1],
        rowval : vec![1],
        nzval  : vec![2.]};

    let q = vec![0., 0.];

    let A : CscMatrix<f64> =

    CscMatrix{m : 3,
        n : 2,
        colptr : vec![0,1,2],
        rowval : vec![1,2],
        nzval  : vec![-2.,-1.]};

    let b = vec![1., -2., -2.];


    (P,q,A,b)
}


fn main() {

    let (P,q,A,b) = _problem_data();

    let cone_types = [SecondOrderConeT];

    let cone_dims  = [3];

    let settings = SettingsBuilder::default()
            .equilibrate_enable(false)
            .max_iter(50)
            .verbose(true)
            .build().unwrap();

    //PJG: no borrow on settings sucks here
    let mut solver = DefaultSolver::
        new(&P,&q,&A,&b,&cone_types,&cone_dims, settings);

    solver.solve();
    
}
