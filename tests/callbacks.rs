#![allow(non_snake_case)]

#[cfg(test)]
mod callback_test {

    use clarabel::solver::default::ffi::DefaultInfoFFI;
    use clarabel::solver::DefaultInfo;
    use clarabel::{algebra::*, solver::*};
    use core::ffi::{c_int, c_void};

    // setup a custom termination function
    fn callback_r(info: &DefaultInfo<f64>) -> bool {
        if info.iterations < 3 {
            false //continue
        } else {
            true // stop
        }
    }

    #[no_mangle]
    pub extern "C" fn callback_fcn(
        _info: *const DefaultInfoFFI<f64>,
        _data_ptr: *mut c_void,
    ) -> c_int {
        // just terminate immediately
        1
    }

    // First, define a struct to hold your state
    struct CallbackState {
        counter: c_int,
    }
    impl CallbackState {
        fn new() -> Self {
            CallbackState { counter: -1 }
        }
    }

    #[no_mangle]
    pub extern "C" fn callback_with_state_fcn(
        _info: *const DefaultInfoFFI<f64>,
        data_ptr: *mut c_void,
    ) -> c_int {
        let state = unsafe { &mut *(data_ptr as *mut CallbackState) };
        state.counter += 1;
        if state.counter >= 3 {
            1 // terminate
        } else {
            0 // continue
        }
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
        solver.set_termination_callback_c(callback_fcn, std::ptr::null_mut());
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 0);
    }

    #[test]
    fn test_callbacks_with_state() {
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
        solver.set_termination_callback_c(callback_fcn, std::ptr::null_mut());
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 0);

        // turn it on with a stateful C implemented callback
        let state = &mut CallbackState::new();
        let ptr = state as *mut _ as *mut c_void;
        solver.set_termination_callback_c(callback_with_state_fcn, ptr);
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::CallbackTerminated);
        assert_eq!(solver.solution.iterations, 3);
    }
}
