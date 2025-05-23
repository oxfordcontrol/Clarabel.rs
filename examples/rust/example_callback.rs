#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

fn main() {
    // This example shows how to use a custom termination function
    // to halt the solver

    let P: CscMatrix<f64> = CscMatrix::<f64>::zeros((2, 2));
    let q = vec![1., -1.];

    //a 2-d box constraint, separated into 4 inequalities.
    //A = [I; -I]
    let _A = CscMatrix::new(
        4,                      // m
        2,                      // n
        vec![0, 2, 4],          // colptr
        vec![0, 2, 1, 3],       // rowval
        vec![1., -1., 1., -1.], // nzval
    );

    // easier way - use the From trait to construct A:
    let A = CscMatrix::from(&[
        [1., 0.],  //
        [0., 1.],  //
        [-1., 0.], //
        [0., -1.], //
    ]);

    let b = vec![1.; 4];

    let cones = [NonnegativeConeT(4)];

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

    // setup a custom termination function
    let f = |info: &DefaultInfo<f64>| {
        if info.iterations < 3 {
            println!("tick");
            false //continue
        } else {
            println!("BOOM!\n");
            true // stop
        }
    };

    solver.set_termination_callback(f);
    solver.solve();

    // setup a custom termination function that carries internal state
    let mut counter = 0;
    let fmut = move |info: &DefaultInfo<f64>| {
        counter += 1;
        println!("Iteration {}: counter = {}", info.iterations, counter);
        if counter < 3 {
            false // continue
        } else {
            println!("BOOM!\n");
            true // stop
        }
    };

    solver.set_termination_callback(fmut);
    solver.solve();

    // turn it off and run again
    solver.unset_termination_callback();
    solver.solve();
}
