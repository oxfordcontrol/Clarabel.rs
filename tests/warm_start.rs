#![allow(non_snake_case)]

use clarabel::{algebra::*, solver::*, solver::traits::{Settings, Variables}};

#[allow(clippy::type_complexity)]
fn warm_start_test_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    // Simple QP: min 0.5 x'Px + q'x  s.t. Ax + s = b, s in K
    let P = CscMatrix::from(&[
        [4., 1.], //
        [1., 2.], //
    ]);
    let q = vec![1., 1.];

    let A = CscMatrix::from(&[
        [1., 1.],  //
        [1., 0.],  //
        [0., 1.],  //
        [-1., 0.], //
        [0., -1.], //
    ]);
    let b = vec![1., 0.7, 0.7, 0., 0.];

    let cones = vec![NonnegativeConeT(5)];

    (P, q, A, b, cones)
}

#[test]
fn test_warm_start_same_problem() {
    let (P, q, A, b, cones) = warm_start_test_data();

    let settings = DefaultSettingsBuilder::default()
        .presolve_enable(false)
        .verbose(false)
        .build()
        .unwrap();

    // Cold-start solve
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let cold_iterations = solver.solution.iterations;
    let cold_x = solver.solution.x.clone();

    // Restore internal-form variables cached before unscaling
    solver.variables.copy_from(&solver.prev_vars);

    // Enable warm-start via convenience method
    solver.set_warm_start_skip(true);

    // Warm-start solve (same problem, starting from optimal point)
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    // Starting from the optimal point should converge faster
    assert!(
        solver.solution.iterations <= cold_iterations,
        "warm-start iterations ({}) should be <= cold-start iterations ({})",
        solver.solution.iterations,
        cold_iterations
    );

    // Solutions should match
    assert!(
        solver.solution.x.dist(&cold_x) <= 1e-6,
        "warm-start solution should match cold-start solution"
    );
}

#[test]
fn test_warm_start_perturbed_problem() {
    let (P, q, A, b, cones) = warm_start_test_data();

    let settings = DefaultSettingsBuilder::default()
        .presolve_enable(false)
        .verbose(false)
        .build()
        .unwrap();

    // Cold-start solve of original problem
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone()).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let cold_iterations = solver.solution.iterations;

    // Restore internal-form variables
    solver.variables.copy_from(&solver.prev_vars);

    // Small perturbation to q.  The perturbation must be small enough
    // that the initial dual residual (proportional to the perturbation)
    // can converge below tol_feas before μ hits machine precision.
    let q2 = vec![1.001, 0.999];
    solver.update_q(&q2).unwrap();

    // Enable warm-start via convenience method
    solver.set_warm_start_skip(true);

    // Warm-start solve of perturbed problem
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let warm_x = solver.solution.x.clone();

    // Warm-start should converge faster than cold-start
    assert!(
        solver.solution.iterations <= cold_iterations,
        "warm-start iterations ({}) should be <= cold-start iterations ({})",
        solver.solution.iterations,
        cold_iterations
    );

    // Cold-start solve of the same perturbed problem for comparison
    let mut solver_cold = DefaultSolver::new(&P, &q2, &A, &b, &cones, settings).unwrap();
    solver_cold.solve();
    assert_eq!(solver_cold.solution.status, SolverStatus::Solved);

    // Both should arrive at the same solution
    assert!(
        warm_x.dist(&solver_cold.solution.x) <= 1e-6,
        "warm-start and cold-start solutions should match for perturbed problem"
    );
}

#[test]
fn test_warm_start_skip_default_false() {
    let settings = DefaultSettings::<f64>::default();
    assert!(
        !settings.warm_start_skip,
        "warm_start_skip should default to false"
    );
}

#[test]
fn test_set_warm_start_skip() {
    let (P, q, A, b, cones) = warm_start_test_data();

    let settings = DefaultSettingsBuilder::default()
        .presolve_enable(false)
        .verbose(false)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

    // Verify default is off
    assert!(!solver.settings().core().warm_start_skip);

    // Toggle on
    solver.set_warm_start_skip(true);
    assert!(solver.settings().core().warm_start_skip);

    // Toggle back off
    solver.set_warm_start_skip(false);
    assert!(!solver.settings().core().warm_start_skip);
}

#[test]
fn test_cold_start_after_warm_start() {
    // Verify that disabling warm_start_skip restores normal cold-start behavior
    let (P, q, A, b, cones) = warm_start_test_data();

    let settings = DefaultSettingsBuilder::default()
        .presolve_enable(false)
        .verbose(false)
        .build()
        .unwrap();

    // First: cold solve
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone()).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let first_cold_x = solver.solution.x.clone();
    let first_cold_iters = solver.solution.iterations;

    // Second: warm-start re-solve
    solver.variables.copy_from(&solver.prev_vars);
    solver.set_warm_start_skip(true);
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(solver.solution.iterations <= first_cold_iters);

    // Third: disable warm-start, solve again (should behave like cold-start)
    solver.set_warm_start_skip(false);
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert_eq!(solver.solution.iterations, first_cold_iters);
    assert!(solver.solution.x.dist(&first_cold_x) <= 1e-6);
}

#[test]
fn test_warm_start_multiple_resolves() {
    // Warm-start through a sequence of small perturbations
    let (P, q, A, b, cones) = warm_start_test_data();

    let settings = DefaultSettingsBuilder::default()
        .presolve_enable(false)
        .verbose(false)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone()).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let cold_iterations = solver.solution.iterations;

    // Solve a sequence of slightly perturbed problems
    for i in 1..=5 {
        let delta = 0.001 * i as f64;
        let q_new = vec![1.0 + delta, 1.0 - delta];

        solver.variables.copy_from(&solver.prev_vars);
        solver.update_q(&q_new).unwrap();
        solver.set_warm_start_skip(true);
        solver.solve();

        assert_eq!(
            solver.solution.status,
            SolverStatus::Solved,
            "failed at perturbation step {i}"
        );
        assert!(
            solver.solution.iterations <= cold_iterations,
            "step {i}: warm-start iterations ({}) should be <= cold-start ({})",
            solver.solution.iterations,
            cold_iterations
        );

        // Verify against fresh cold-start
        let mut solver_ref =
            DefaultSolver::new(&P, &q_new, &A, &b, &cones, settings.clone()).unwrap();
        solver_ref.solve();
        assert!(
            solver.solution.x.dist(&solver_ref.solution.x) <= 1e-6,
            "step {i}: warm-start solution diverged from cold-start"
        );
    }
}
