use clarabel::solver::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Example: Custom Variable Initialization");
    println!("======================================");
    println!("Note: This example shows the API but the initialize_with_values");
    println!("function is mainly used internally by the solver.");
    println!("For warm starting, use the solver's warm start settings instead.");
    println!();
    
    println!("The initialize_with_values function provides:");
    println!("- Flexible partial initialization");
    println!("- Automatic validation of constraints (s > 0, z > 0, τ > 0, κ > 0)");
    println!("- Dimension checking");
    println!("- Clear error reporting");
    println!("- Cone-aware validation for different constraint types");
    println!();
    
    println!("For practical warm starting, see:");
    println!("- example_warm_start.rs (Rust)");
    println!("- example_warm_start_py.py (Python)");

    Ok(())
}
