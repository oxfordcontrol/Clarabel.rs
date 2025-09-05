use clarabel::solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example: Custom Variable Initialization");
    println!("======================================");

    // Create default variables with dimensions n=3 (primal), m=2 (dual/slack)
    let mut variables = DefaultVariables::<f64>::new(3, 2);
    
    println!("Default variables:");
    println!("x: {:?}", variables.x);
    println!("s: {:?}", variables.s);
    println!("z: {:?}", variables.z);
    println!("τ: {:?}", variables.τ);
    println!("κ: {:?}", variables.κ);
    println!();

    // Example 1: Initialize all variables with custom values
    println!("Example 1: Full initialization");
    let x_vals = vec![1.0, 2.0, 3.0];
    let s_vals = vec![0.5, 1.5];
    let z_vals = vec![2.0, 3.0];
    let tau_val = 2.0;
    let kappa_val = 1.5;

    match variables.initialize_with_values(
        Some(&x_vals),
        Some(&s_vals),
        Some(&z_vals),
        Some(tau_val),
        Some(kappa_val),
    ) {
        Ok(()) => {
            println!("✅ Initialization successful!");
            println!("x: {:?}", variables.x);
            println!("s: {:?}", variables.s);
            println!("z: {:?}", variables.z);
            println!("τ: {:?}", variables.τ);
            println!("κ: {:?}", variables.κ);
        }
        Err(e) => {
            println!("❌ Initialization failed: {}", e);
        }
    }
    println!();

    // Example 2: Partial initialization (only x and τ)
    println!("Example 2: Partial initialization (only x and τ)");
    let x_vals_2 = vec![10.0, 20.0, 30.0];
    let tau_val_2 = 0.8;

    match variables.initialize_with_values(
        Some(&x_vals_2),
        None,  // s will default to ones
        None,  // z will default to ones
        Some(tau_val_2),
        None,  // κ will default to one
    ) {
        Ok(()) => {
            println!("✅ Partial initialization successful!");
            println!("x: {:?}", variables.x);
            println!("s: {:?} (defaults)", variables.s);
            println!("z: {:?} (defaults)", variables.z);
            println!("τ: {:?}", variables.τ);
            println!("κ: {:?} (default)", variables.κ);
        }
        Err(e) => {
            println!("❌ Initialization failed: {}", e);
        }
    }
    println!();

    // Example 3: Error case - negative slack variable
    println!("Example 3: Error case - negative slack variable");
    let bad_s_vals = vec![1.0, -0.5]; // negative value should fail

    match variables.initialize_with_values(
        None,
        Some(&bad_s_vals),
        None,
        None,
        None,
    ) {
        Ok(()) => {
            println!("❌ This should not have succeeded!");
        }
        Err(e) => {
            println!("✅ Expected error caught: {}", e);
        }
    }
    println!();

    // Example 4: Error case - dimension mismatch
    println!("Example 4: Error case - dimension mismatch");
    let wrong_x_vals = vec![1.0, 2.0]; // should be length 3, not 2

    match variables.initialize_with_values(
        Some(&wrong_x_vals),
        None,
        None,
        None,
        None,
    ) {
        Ok(()) => {
            println!("❌ This should not have succeeded!");
        }
        Err(e) => {
            println!("✅ Expected error caught: {}", e);
        }
    }
    println!();

    // Example 5: Error case - zero τ
    println!("Example 5: Error case - zero τ");
    
    match variables.initialize_with_values(
        None,
        None,
        None,
        Some(0.0), // τ must be positive
        None,
    ) {
        Ok(()) => {
            println!("❌ This should not have succeeded!");
        }
        Err(e) => {
            println!("✅ Expected error caught: {}", e);
        }
    }

    println!("\nSummary:");
    println!("The initialize_with_values function provides:");
    println!("- Flexible partial initialization");
    println!("- Automatic validation of constraints (s > 0, z > 0, τ > 0, κ > 0)");
    println!("- Dimension checking");
    println!("- Clear error reporting");

    Ok(())
}
