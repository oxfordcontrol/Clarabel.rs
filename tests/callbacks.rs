#![allow(non_snake_case)]

#[cfg(test)]
mod callback_test {

    use clarabel::solver::default::ffi::DefaultInfoFFI;
    use clarabel::solver::DefaultInfo;
    use clarabel::{algebra::*, solver::*};
    use core::ffi::c_int;

    // setup a custom termination function
    fn callback_r(info: &DefaultInfo<f64>) -> bool {
        if info.iterations < 3 {
            println!("tick");
            false //continue
        } else {
            println!("BOOM!\n");
            true // stop
        }
    }

    #[no_mangle]
    pub extern "C" fn callback_c(_info: *const DefaultInfoFFI<f64>) -> c_int {
        // just terminate immediately
        1
    }

    #[test]
    fn test_callbacks() {
        let P = CscMatrix::identity(1);
        let c = [0.];
        let A = CscMatrix::identity(1);
        let b = [1.];
        let cones = [NonnegativeConeT(1)];

        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

        solver.set_termination_callback(callback_r);
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 3);

        // turn it off and run again
        solver.unset_termination_callback();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);

        // turn it back on with a C implemented callback
        solver.set_termination_callback_c(callback_c);
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 0);
    }
}
