#![allow(non_snake_case)]
use clarabel::algebra::*;
use clarabel::solver::*;

fn main() {
    println!("Warm Start Example");
    println!("==================");
    
    // Define a simple QP problem: minimize 0.5*x'*P*x + q'*x subject to A*x = b, x >= 0
    
    let P = CscMatrix::from(&[
        [6., 0.], //
        [0., 4.], //
    ]);

    let q = vec![-1., -4.];

    let A = CscMatrix::from(&[
        [1., -2.], // equality constraint
        [1., 0.],  // inequality constraint
        [0., 1.],  // inequality constraint
        [-1., 0.], // inequality constraint (lower bound)
        [0., -1.], // inequality constraint (lower bound)
    ]);

    let b = vec![0., 1., 1., 1., 1.];
    let cones = [ZeroConeT(1), NonnegativeConeT(4)];

    println!("Problem definition:");
    println!("P = {:?}", P);
    println!("q = {:?}", q);
    println!("A = {:?}", A);
    println!("b = {:?}", b);
    println!("cones = {:?}", cones);
    println!();

    // First solve: without warm start
    println!("=== First solve: Cold start (default initialization) ===");
    let settings = DefaultSettingsBuilder::default()
        .verbose(true)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    let cold_start_solution = solver.solution.clone();
    let cold_start_iterations = solver.solution.iterations;
    
    println!("Cold start solution:");
    println!("x = {:?}", cold_start_solution.x);
    println!("z = {:?}", cold_start_solution.z);
    println!("s = {:?}", cold_start_solution.s);
    println!("Iterations: {}", cold_start_iterations);
    println!();

    // Second solve: with warm start using the solution from the first solve
    println!("=== Second solve: Warm start (using previous solution) ===");
    
    // Create a slightly perturbed version of the solution to simulate a warm start scenario
    let warm_x = cold_start_solution.x.iter().map(|&x| x * 0.9).collect::<Vec<_>>();
    let warm_s = cold_start_solution.s.iter().map(|&s| f64::max(s + 0.1, 0.1)).collect::<Vec<_>>();
    let warm_z = cold_start_solution.z.iter().map(|&z| f64::max(z + 0.1, 0.1)).collect::<Vec<_>>();
    let warm_tau = 0.8;
    let warm_kappa = 1.2;

    println!("Warm start values:");
    println!("x = {:?}", warm_x);
    println!("s = {:?}", warm_s);
    println!("z = {:?}", warm_z);
    println!("τ = {}", warm_tau);
    println!("κ = {}", warm_kappa);
    println!();

    let warm_settings = DefaultSettingsBuilder::default()
        .verbose(true)
        .warm_start_enable(true)
        .warm_start_x(Some(warm_x))
        .warm_start_s(Some(warm_s))
        .warm_start_z(Some(warm_z))
        .warm_start_tau(Some(warm_tau))
        .warm_start_kappa(Some(warm_kappa))
        .build()
        .unwrap();

    let mut warm_solver = DefaultSolver::new(&P, &q, &A, &b, &cones, warm_settings).unwrap();
    warm_solver.solve();

    let warm_start_solution = warm_solver.solution.clone();
    let warm_start_iterations = warm_solver.solution.iterations;
    
    println!("Warm start solution:");
    println!("x = {:?}", warm_start_solution.x);
    println!("z = {:?}", warm_start_solution.z);
    println!("s = {:?}", warm_start_solution.s);
    println!("Iterations: {}", warm_start_iterations);
    println!();

    // Third solve: with warm start enabled but no values provided (should fall back to default)
    println!("=== Third solve: Warm start enabled but no values (fallback to default) ===");
    
    let fallback_settings = DefaultSettingsBuilder::default()
        .verbose(true)
        .warm_start_enable(true)
        // No warm start values provided - should fall back to default initialization
        .build()
        .unwrap();

    let mut fallback_solver = DefaultSolver::new(&P, &q, &A, &b, &cones, fallback_settings).unwrap();
    fallback_solver.solve();

    let fallback_iterations = fallback_solver.solution.iterations;
    
    println!("Fallback solution iterations: {}", fallback_iterations);
    println!();

    // Fourth solve: with invalid warm start values (should fall back to default)
    println!("=== Fourth solve: Invalid warm start values (should fallback) ===");
    
    let invalid_s = vec![-1.0, 0.5]; // negative value should cause fallback
    
    let invalid_settings = DefaultSettingsBuilder::default()
        .verbose(false) // Reduce output for this test
        .warm_start_enable(true)
        .warm_start_s(Some(invalid_s))
        .build()
        .unwrap();

    let mut invalid_solver = DefaultSolver::new(&P, &q, &A, &b, &cones, invalid_settings).unwrap();
    invalid_solver.solve();

    let invalid_iterations = invalid_solver.solution.iterations;
    
    println!("Invalid warm start (fallback) iterations: {}", invalid_iterations);
    println!();

    // Comparison summary
    println!("=== SUMMARY ===");
    println!("Cold start iterations:        {}", cold_start_iterations);
    println!("Warm start iterations:        {}", warm_start_iterations);
    println!("Fallback iterations:          {}", fallback_iterations);
    println!("Invalid warmstart iterations: {}", invalid_iterations);
    
    if warm_start_iterations <= cold_start_iterations {
        println!("✅ Warm start improved performance!");
    } else {
        println!("ℹ️  In this simple case, warm start didn't provide improvement (this is normal for simple problems)");
    }
    
    println!();
    println!("Features demonstrated:");
    println!("- ✅ Warm start with custom initial values");
    println!("- ✅ Automatic validation of warm start values");
    println!("- ✅ Graceful fallback to default initialization on failure");
    println!("- ✅ Partial warm start (not all values need to be provided)");
}
