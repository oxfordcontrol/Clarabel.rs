use clarabel::algebra::*;
use clarabel::solver::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn solve_problem() -> Result<(), JsValue> {
    let (P, q, A, b) = problem_data();

    let cones = [NonnegativeConeT(b.len())];

    let settings = DefaultSettingsBuilder::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .build()
        .map_err(|e| JsValue::from_str(&format!("Settings error: {:?}", e)))?;

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);

    solver.solve();

    Ok(())
}

fn problem_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let n = 2000000;

    let P = CscMatrix::identity(n);

    // construct A = [I; -I]
    let I1 = CscMatrix::<f64>::identity(n);
    let mut I2 = CscMatrix::<f64>::identity(n);
    I2.negate();

    let A = CscMatrix::vcat(&I1, &I2);

    let q = vec![1.; n];
    let b = vec![1.; 2 * n];

    (P, q, A, b)
}
