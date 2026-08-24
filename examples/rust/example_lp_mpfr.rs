#![allow(non_snake_case)]
//! LP example solved end-to-end with the MPFR-precision backend.
//!
//! Solves the same 2-D box LP as `example_lp.rs`, but with
//! `T = MpfrFloat` at 167-bit (~50 dps) working precision. Every
//! arithmetic op rounds to MPFR's native correctly-rounded result;
//! denominators are bounded by the working precision (unlike the
//! exact-rational backend's geometric blowup). For QOU's H_n
//! confinement work this is the recommended high-precision path —
//! same precision target as `seminormal_mpfr.rs`.
//!
//! Run:
//!   cargo run --no-default-features --features serde,mpfr \
//!             --example lp_mpfr --release

use clarabel::algebra::*;
use clarabel::solver::*;
use num_traits::FromPrimitive;

type T = MpfrFloat;

fn rat(n: i64) -> T {
    T::from_i64(n).unwrap()
}

fn main() {
    // 167 bits ≈ 50 decimal digits, matching QOU R5_FULL_PLAN.md.
    set_mpfr_default_precision(167);

    // P = 0 (LP)
    let P: CscMatrix<T> = CscMatrix::<T>::zeros((2, 2));

    // q = [1, -1]
    let q: Vec<T> = vec![rat(1), -rat(1)];

    // A = [I; -I]  (2-d box separated into 4 inequalities)
    let one: T = rat(1);
    let neg_one: T = -rat(1);
    let A = CscMatrix::new(
        4,                                     // m
        2,                                     // n
        vec![0, 2, 4],                         // colptr
        vec![0, 2, 1, 3],                      // rowval
        vec![one.clone(), neg_one.clone(), one, neg_one], // nzval
    );

    let b: Vec<T> = vec![rat(1); 4];

    let cones = [NonnegativeConeT(4)];

    // Tolerance: 1e-12, well below f64's typical 1e-8 default but
    // above the static-regularization floor (~1e-13 from
    // settings.static_regularization_eps); useful as a "tighter than
    // f64 default but still reachable" middle ground.
    let tol = T::from_f64(1e-12).unwrap();

    let settings = DefaultSettingsBuilder::<T>::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .tol_feas(tol.clone())
        .tol_gap_abs(tol.clone())
        .tol_gap_rel(tol)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

    solver.solve();

    // Extract the primal solution at full MPFR precision.
    println!();
    println!("=== MPFR primal solution (167-bit working precision) ===");
    for (i, xi) in solver.solution.x.iter().enumerate() {
        println!("  x[{i}] = {}", xi);
        println!("    (≈ {} in f64)", xi.to_f64());
    }

    let obj = &solver.solution.obj_val;
    println!("  obj  = {}", obj);
    println!("    (≈ {} in f64)", obj.to_f64());

    // Per-iteration trace: max precision-bits used across the primal
    // x vector. For MpfrFloat this is constant at the working precision
    // (167) since every op rounds to that — useful primarily as a
    // sanity check that the configured precision is actually used.
    let trace = &solver.info.iter_diagnostics;
    println!();
    println!("Per-iteration mantissa-bits trace (max across primal x):");
    for d in trace {
        println!("  iter {}: numer_bits = {}", d.iter, d.max_numer_bits);
    }
}
