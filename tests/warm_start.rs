#![allow(non_snake_case)]

#[cfg(test)]
mod warm_start_tests {
    use clarabel::algebra::*;
    use clarabel::solver::*;

    #[test]
    fn test_warm_start_basic() {
        let P = CscMatrix::from(&[[6., 0.], [0., 4.]]);
        let q = vec![-1., -4.];
        let A = CscMatrix::from(&[
            [1., -2.],
            [1., 0.],
            [0., 1.],
            [-1., 0.],
            [0., -1.],
        ]);
        let b = vec![0., 1., 1., 1., 1.];
        let cones = [ZeroConeT(1), NonnegativeConeT(4)];

        // Test that warm start settings can be created and used
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .warm_start_enable(true)
            .warm_start_x(Some(vec![0.4, 0.2]))
            .warm_start_s(Some(vec![0.1, 0.6, 0.8, 1.4, 1.2]))
            .warm_start_z(Some(vec![0.1, 0.1, 0.1, 0.1, 0.1]))
            .warm_start_tau(Some(0.9))
            .warm_start_kappa(Some(1.1))
            .build()
            .unwrap();

        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();

        // Verify that the solver completes successfully
        assert!(matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ));
        
        // Verify that warm start was used
        assert!(solver.solution.warm_start_used, "Solution should indicate warm start was used");
        assert!(solver.info.warm_start_used, "Info should indicate warm start was used");
    }

    #[test]
    fn test_warm_start_partial() {
        let P = CscMatrix::from(&[[6., 0.], [0., 4.]]);
        let q = vec![-1., -4.];
        let A = CscMatrix::from(&[
            [1., -2.],
            [1., 0.],
            [0., 1.],
            [-1., 0.],
            [0., -1.],
        ]);
        let b = vec![0., 1., 1., 1., 1.];
        let cones = [ZeroConeT(1), NonnegativeConeT(4)];

        // Test partial warm start (only x and tau)
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .warm_start_enable(true)
            .warm_start_x(Some(vec![0.4, 0.2]))
            .warm_start_tau(Some(0.9))
            // Other values should default
            .build()
            .unwrap();

        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();

        assert!(matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ));
    }

    #[test]
    fn test_warm_start_disabled() {
        let P = CscMatrix::from(&[[6., 0.], [0., 4.]]);
        let q = vec![-1., -4.];
        let A = CscMatrix::from(&[
            [1., -2.],
            [1., 0.],
            [0., 1.],
            [-1., 0.],
            [0., -1.],
        ]);
        let b = vec![0., 1., 1., 1., 1.];
        let cones = [ZeroConeT(1), NonnegativeConeT(4)];

        // Test with warm start disabled (should use default initialization)
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .warm_start_enable(false)
            .warm_start_x(Some(vec![0.4, 0.2])) // This should be ignored
            .build()
            .unwrap();

        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();

        assert!(matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ));
    }

    #[test]
    fn test_warm_start_invalid_values_fallback() {
        let P = CscMatrix::from(&[[6., 0.], [0., 4.]]);
        let q = vec![-1., -4.];
        let A = CscMatrix::from(&[
            [1., -2.],
            [1., 0.],
            [0., 1.],
            [-1., 0.],
            [0., -1.],
        ]);
        let b = vec![0., 1., 1., 1., 1.];
        let cones = [ZeroConeT(1), NonnegativeConeT(4)];

        // Test with invalid warm start values (negative s values)
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .warm_start_enable(true)
            .warm_start_s(Some(vec![-1.0, 0.6, 0.8, 1.4, 1.2])) // Invalid: negative value
            .build()
            .unwrap();

        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();

        // Should still solve successfully by falling back to default initialization
        assert!(matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ));
    }

    #[test]
    fn test_warm_start_dimension_mismatch_fallback() {
        let P = CscMatrix::from(&[[6., 0.], [0., 4.]]);
        let q = vec![-1., -4.];
        let A = CscMatrix::from(&[
            [1., -2.],
            [1., 0.],
            [0., 1.],
            [-1., 0.],
            [0., -1.],
        ]);
        let b = vec![0., 1., 1., 1., 1.];
        let cones = [ZeroConeT(1), NonnegativeConeT(4)];

        // Test with wrong dimensions (x should be length 2, not 3)
        let settings = DefaultSettingsBuilder::default()
            .verbose(false)
            .warm_start_enable(true)
            .warm_start_x(Some(vec![0.4, 0.2, 0.1])) // Wrong dimension!
            .build()
            .unwrap();

        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();

        // Should still solve successfully by falling back to default initialization
        assert!(matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ));
        
        // Should report that warm start was NOT used due to validation failure
        assert!(!solver.solution.warm_start_used, "Solution should indicate warm start was NOT used due to validation failure");
        assert!(!solver.info.warm_start_used, "Info should indicate warm start was NOT used due to validation failure");
    }

    #[test]
    fn test_warm_start_status_reporting() {
        let P = CscMatrix::from(&[[6., 0.], [0., 4.]]);
        let q = vec![-1., -4.];
        let A = CscMatrix::from(&[
            [1., -2.],
            [1., 0.],
            [0., 1.],
            [-1., 0.],
            [0., -1.],
        ]);
        let b = vec![0., 1., 1., 1., 1.];
        let cones = [ZeroConeT(1), NonnegativeConeT(4)];

        // Test warm start enabled - should report warm start was used
        let settings_warm = DefaultSettingsBuilder::default()
            .verbose(false)
            .warm_start_enable(true)
            .warm_start_x(Some(vec![0.4, 0.2]))
            .warm_start_tau(Some(0.9))
            .build()
            .unwrap();

        let mut solver_warm = DefaultSolver::new(&P, &q, &A, &b, &cones, settings_warm).unwrap();
        solver_warm.solve();
        
        assert!(solver_warm.solution.warm_start_used, "Solution should indicate warm start was used");
        assert!(solver_warm.info.warm_start_used, "Info should indicate warm start was used");

        // Test warm start disabled - should report warm start was NOT used
        let settings_cold = DefaultSettingsBuilder::default()
            .verbose(false)
            .warm_start_enable(false)
            .build()
            .unwrap();

        let mut solver_cold = DefaultSolver::new(&P, &q, &A, &b, &cones, settings_cold).unwrap();
        solver_cold.solve();
        
        assert!(!solver_cold.solution.warm_start_used, "Solution should indicate warm start was NOT used");
        assert!(!solver_cold.info.warm_start_used, "Info should indicate warm start was NOT used");
    }
}
