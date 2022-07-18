#![allow(non_snake_case)]

use clarabel::core::*;
use clarabel::implementations::default::*;

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

    let cones = [SecondOrderConeT(3)];

    let settings = DefaultSettingsBuilder::default()
            .equilibrate_enable(false)
            .max_iter(50)
            .verbose(true)
            .build().unwrap();

    let mut solver = DefaultSolver::
        new(&P,&q,&A,&b,&cones, settings);

    solver.solve();
    
}
