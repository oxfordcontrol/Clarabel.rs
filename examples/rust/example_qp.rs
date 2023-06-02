#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

fn main() {
    // QP Example

    // let P = CscMatrix::identity(2);    // For P = I
    // let P = CscMatrix::zeros((2,2));   // For P = 0

    // direct from sparse data
    let _P = CscMatrix::new(
        2,             // m
        2,             // n
        vec![0, 1, 2], // colptr
        vec![0, 1],    // rowval
        vec![6., 4.],  // nzval
    );

    // or an easier way for small problems...
    let P = CscMatrix::from(&[
        [6., 0.], //
        [0., 4.], //
    ]);

    let q = vec![-1., -4.];

    //direct from sparse data
    let _A = CscMatrix::new(
        5,                               // m
        2,                               // n
        vec![0, 3, 6],                   // colptr
        vec![0, 1, 3, 0, 2, 4],          // rowval
        vec![1., 1., -1., -2., 1., -1.], // nzval
    );

    // or an easier way for small problems...
    let A = CscMatrix::from(&[
        [1., -2.], // <-- LHS of equality constraint (lower bound)
        [1., 0.],  // <-- LHS of inequality constraint (upper bound)
        [0., 1.],  // <-- LHS of inequality constraint (upper bound)
        [-1., 0.], // <-- LHS of inequality constraint (lower bound)
        [0., -1.], // <-- LHS of inequality constraint (lower bound)
    ]);

    let b = vec![0., 1., 1., 1., 1.];

    let cones = [ZeroConeT(1), NonnegativeConeT(4)];

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();

    println!("Solution(x)     = {:?}", solver.solution.x);
    println!("Multipliers (z) = {:?}", solver.solution.z);
    println!("Slacks (s)      = {:?}", solver.solution.s);
}
