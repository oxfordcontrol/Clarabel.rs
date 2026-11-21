#![allow(non_snake_case)]
//! LP example solved end-to-end with the exact-rational backend.
//!
//! Solves the same 2-D box LP as `example_lp.rs`, but with
//! `T = RationalReal`. The primal iterates and the final solution are
//! exact rationals; only the per-iteration print path rounds (via
//! `LowerExp` -> `f64`) for legibility.
//!
//! Run:
//!   cargo run --no-default-features --features serde,bigrational \
//!             --example lp_rational --release
//!
//! # Two precision modes
//!
//! The example below sets `set_max_arena_bits(Some(256))`, which engages
//! the inner-loop precision-capping mode: every arithmetic op rounds its
//! result to `m / 2^256` if it would otherwise exceed 256 numerator or
//! denominator bits. This bounds the per-op cost so the IPM runs in
//! seconds rather than minutes. Precision is still ~77 decimal digits —
//! 5× f64 — so the 1e-8 tolerance set below is met with massive margin.
//!
//! To see the unbounded **exact** mode (which is the headline guarantee
//! of the `bigrational` backend — `1/3 + 1/3 + 1/3 == 1` exactly),
//! comment out the `set_max_arena_bits` line. Expect many minutes per
//! IPM iteration on this problem because BigRational denominators grow
//! geometrically without the cap; this is intrinsic to exact rational
//! LP solving (practical exact-LP packages mitigate via iterative
//! refinement or specialized rational simplex methods, neither of
//! which Clarabel does).
//!
//! For QOU-scale workloads the planned MPFR backend (Phase 8) will give
//! high-precision floats with bounded denominators in a more direct
//! way. The `bigrational` backend with `max_arena_bits` is the
//! "high-precision rational" middle ground: rationals when they fit,
//! rounded-to-precision-p when they don't.

use clarabel::algebra::*;
use clarabel::solver::*;
use num_bigint::BigInt;

type T = RationalReal;

fn rat_from_int(n: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(1))
}

fn rat_from_pair(n: i64, d: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(d))
}

fn main() {
    // Set the working precision used by transcendentals (sqrt etc).
    // Pure LP doesn't actually consume this — only barrier transcendentals
    // do — but it's the right place to demonstrate the knob.
    set_precision_bits(128);

    // Engage inner-loop precision capping so the IPM runs in seconds,
    // not minutes. At 256 bits ≈ 77 decimal digits, the cap is ~5×
    // f64's precision — far in excess of the 1e-8 tolerance we use
    // below — but bounded enough that BigRational arithmetic doesn't
    // blow up geometrically. Comment this out to see the unbounded
    // exact-mode behaviour (correct but minutes per iteration).
    set_max_arena_bits(Some(256));

    // Reset the per-thread arena so the example doesn't inherit any
    // state from a prior invocation.
    reset_arena();

    // P = 0 (LP)
    let P: CscMatrix<T> = CscMatrix::<T>::zeros((2, 2));

    // q = [1, -1]
    let q: Vec<T> = vec![rat_from_int(1), -rat_from_int(1)];

    // A = [I; -I]  (2-d box separated into 4 inequalities)
    // Build A by hand with rational entries.
    let one: T = rat_from_int(1);
    let neg_one: T = -rat_from_int(1);
    let A = CscMatrix::new(
        4,                                     // m
        2,                                     // n
        vec![0, 2, 4],                         // colptr
        vec![0, 2, 1, 3],                      // rowval
        vec![one, neg_one, one, neg_one],      // nzval
    );

    let b: Vec<T> = vec![rat_from_int(1); 4];

    let cones = [NonnegativeConeT(4)];

    // Settings: the float-typed defaults need RationalReal versions.
    // tol_feas / tol_gap_* are T-typed in DefaultSettings, so we must
    // construct them explicitly. Use 1e-8 as a rational: 1/100_000_000.
    let tol = rat_from_pair(1, 100_000_000);

    let settings = DefaultSettingsBuilder::<T>::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .tol_feas(tol)
        .tol_gap_abs(tol)
        .tol_gap_rel(tol)
        .build()
        .unwrap();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

    solver.solve();

    // Extract the primal solution as exact rationals.
    println!();
    println!("=== Exact-rational primal solution ===");
    for (i, xi) in solver.solution.x.iter().enumerate() {
        let (num, den) = xi.into_pair();
        println!("  x[{i}] = {num} / {den}   (≈ {})", xi.to_f64());
    }

    // Objective value, exact.
    let obj = solver.solution.obj_val;
    let (num, den) = obj.into_pair();
    println!("  obj  = {num} / {den}   (≈ {})", obj.to_f64());

    // Diagnostics: arena size + max numerator/denominator bit-length
    // across the primal solution.
    let max_bits = solver
        .solution
        .x
        .iter()
        .map(|xi| xi.max_bits())
        .max()
        .unwrap_or(0);
    println!();
    println!(
        "Arena size: {} entries; max(numer_bits, denom_bits) across primal x: {}",
        arena_len(),
        max_bits
    );

    // Recover memory before exit (would matter for long-running processes).
    reset_arena();
}
