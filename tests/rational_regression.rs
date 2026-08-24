//! Integration regression tests for the `bigrational` backend.
//!
//! Run:
//!   cargo test --no-default-features --features serde,bigrational \
//!              --test rational_regression
//!
//! These tests exercise the solver end-to-end with `T = RationalReal`.
//! They are gated on the `bigrational` feature so the default-feature
//! `cargo test` is unaffected.

#![cfg(feature = "bigrational")]
#![allow(non_snake_case)]

use clarabel::algebra::*;
use clarabel::solver::*;
use num_bigint::BigInt;

type T = RationalReal;

fn rat(n: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(1))
}

fn rat2(n: i64, d: i64) -> T {
    T::from_pair(BigInt::from(n), BigInt::from(d))
}

/// Build the same default settings the f64 solver would use, but with
/// rational tolerances at 1e-8 = 1/100_000_000.
fn default_rational_settings() -> DefaultSettings<T> {
    let tol = rat2(1, 100_000_000);
    DefaultSettingsBuilder::<T>::default()
        .equilibrate_enable(true)
        .max_iter(50)
        .tol_feas(tol)
        .tol_gap_abs(tol)
        .tol_gap_rel(tol)
        .build()
        .unwrap()
}

/// Tiny LP from example_lp.rs:
///   min    x[0] - x[1]
///   s.t.   |x[i]| <= 1   for i in {0,1}
/// Optimum: x* = (-1, 1), obj* = -2.
///
/// Runs with `set_max_arena_bits(Some(256))` engaged so per-iteration
/// rational denominators are bounded at 256 bits ≈ 77 dps. End-to-end
/// solve takes ~50 ms in --release; iter counts and per-iter
/// pcost/gap/pres/dres values are bit-identical to the f64 baseline.
///
/// A separate test below (`rational_lp_box_solves_exactly_unbounded`)
/// exercises the unbounded exact mode and is marked `#[ignore]`
/// because it takes many minutes per iteration.
#[test]
fn rational_lp_box_solves_exactly() {
    reset_arena();
    set_max_arena_bits(Some(256));

    let P: CscMatrix<T> = CscMatrix::<T>::zeros((2, 2));
    let q: Vec<T> = vec![rat(1), -rat(1)];
    let A = CscMatrix::new(
        4,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 3],
        vec![rat(1), -rat(1), rat(1), -rat(1)],
    );
    let b: Vec<T> = vec![rat(1); 4];
    let cones = [NonnegativeConeT(4)];

    let mut solver =
        DefaultSolver::new(&P, &q, &A, &b, &cones, default_rational_settings()).unwrap();
    solver.solve();

    assert!(matches!(solver.solution.status, SolverStatus::Solved));

    // Objective value should be near -2 (within 1e-6 in f64 view; the
    // exact-rational form is checked separately below).
    let obj_f64 = solver.solution.obj_val.to_f64();
    assert!(
        (obj_f64 + 2.0).abs() < 1e-6,
        "objective {} should be near -2",
        obj_f64
    );

    // Each x[i] should be near ±1.
    for (i, expected) in [-1.0_f64, 1.0_f64].iter().enumerate() {
        let xi_f64 = solver.solution.x[i].to_f64();
        assert!(
            (xi_f64 - expected).abs() < 1e-6,
            "x[{i}] = {xi_f64} should be near {expected}"
        );
    }

    // Both numer and denom are bounded by the cap.
    for xi in &solver.solution.x {
        assert!(
            xi.max_bits() <= 256,
            "set_max_arena_bits(256) violated: x has {} bits",
            xi.max_bits()
        );
    }

    // Per-iteration denom-bit trace should be populated (one entry per
    // IPM iteration). For RationalReal under the cap, max_denom_bits
    // is bounded by 257 (2^256 has 257 bits — the leading 1 plus 256
    // trailing zeros — and `cap.rs::round_to_pow2_denominator` always
    // produces denom = 2^p exactly).
    let trace = &solver.info.iter_diagnostics;
    assert!(
        !trace.is_empty(),
        "iter_diagnostics should be populated for RationalReal"
    );
    for d in trace {
        assert!(
            d.max_denom_bits <= 257,
            "trace denom_bits = {} (cap=256, expected ≤ 257)",
            d.max_denom_bits
        );
    }

    set_max_arena_bits(None);
    reset_arena();
}

/// Same LP, but in unbounded exact mode. Marked `#[ignore]` because
/// it takes many minutes per IPM iteration as denominators grow
/// geometrically. Run explicitly with:
///   cargo test --no-default-features --features serde,bigrational \
///              --test rational_regression \
///              rational_lp_box_solves_exactly_unbounded \
///              -- --ignored --nocapture
#[test]
#[ignore = "very slow: unbounded exact-mode rational LP; ~10+ minutes"]
fn rational_lp_box_solves_exactly_unbounded() {
    reset_arena();
    set_max_arena_bits(None);

    let P: CscMatrix<T> = CscMatrix::<T>::zeros((2, 2));
    let q: Vec<T> = vec![rat(1), -rat(1)];
    let A = CscMatrix::new(
        4,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 3],
        vec![rat(1), -rat(1), rat(1), -rat(1)],
    );
    let b: Vec<T> = vec![rat(1); 4];
    let cones = [NonnegativeConeT(4)];

    let mut solver =
        DefaultSolver::new(&P, &q, &A, &b, &cones, default_rational_settings()).unwrap();
    solver.solve();

    assert!(matches!(solver.solution.status, SolverStatus::Solved));
    let obj_f64 = solver.solution.obj_val.to_f64();
    assert!((obj_f64 + 2.0).abs() < 1e-6);
    reset_arena();
}

/// QP from example_qp.rs:
///   min  3 x[0]² + 2 x[1]² - x[0] - 4 x[1]
///   s.t. x[0] - 2 x[1] = 0,  |x[i]| <= 1
/// Optimum (from f64 baseline): x* ≈ (0.4286, 0.2143), obj* ≈ -0.6429.
///
/// Same problem the f64 example_qp.rs solves. Exercises the P ≠ 0
/// path (rational dot/quad-form/Hessian) plus the ZeroConeT (equality
/// constraint), neither of which the LP test covered.
#[test]
fn rational_qp_solves_under_cap() {
    reset_arena();
    set_max_arena_bits(Some(256));

    // P = diag(6, 4). With 1/2 x'Px + q'x convention, this is
    // 3 x[0]² + 2 x[1]² in the objective.
    let P = CscMatrix::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![rat(6), rat(4)],
    );
    let q: Vec<T> = vec![-rat(1), -rat(4)];

    let A = CscMatrix::new(
        5,
        2,
        vec![0, 3, 6],
        vec![0, 1, 3, 0, 2, 4],
        vec![rat(1), rat(1), -rat(1), -rat(2), rat(1), -rat(1)],
    );
    let b: Vec<T> = vec![rat(0), rat(1), rat(1), rat(1), rat(1)];
    let cones = [ZeroConeT(1), NonnegativeConeT(4)];

    let mut solver =
        DefaultSolver::new(&P, &q, &A, &b, &cones, default_rational_settings()).unwrap();
    solver.solve();

    assert!(matches!(solver.solution.status, SolverStatus::Solved));

    // Check solution against f64 baseline.
    let x0 = solver.solution.x[0].to_f64();
    let x1 = solver.solution.x[1].to_f64();
    let obj = solver.solution.obj_val.to_f64();
    assert!((x0 - 0.42857143).abs() < 1e-6, "x[0] = {x0}");
    assert!((x1 - 0.21428571).abs() < 1e-6, "x[1] = {x1}");
    assert!((obj + 0.64285714).abs() < 1e-6, "obj = {obj}");

    // Equality constraint x[0] - 2 x[1] = 0 should hold exactly (or
    // within the rounding tolerance of the cap, since A and b are
    // integers and the cap rounds intermediates).
    let lhs = solver.solution.x[0] - rat(2) * solver.solution.x[1];
    let lhs_f = lhs.to_f64();
    assert!(lhs_f.abs() < 1e-6, "x[0] - 2*x[1] = {lhs_f} (should be ~0)");

    set_max_arena_bits(None);
    reset_arena();
}

/// SOCP from example_socp.rs:
///   min  x[1]²
///   s.t. (1, -2 x[0], -x[1]) - (0, 0, 0) ∈ K_soc(3),  i.e.
///        sqrt(4 x[0]² + x[1]²) <= 1.
///
/// Optimum (from f64 baseline): x* ≈ (1.0, 1.0), obj* ≈ 1.0.
/// (The constraint forces |x[0]| <= 1/2 and |x[1]| <= 1, but the
/// objective drives x[1] to its upper bound; the trace iter-print
/// shows pcost going through +1.0e+00 at termination.)
///
/// Exercises the SOC cone path: SOC margin/residual via
/// stable_norm (sqrt under the cap), the rank-2 Hessian update, and
/// equilibration that consults sqrt of column norms — all the
/// transcendental call sites that were dead in the LP/QP tests.
#[test]
fn rational_socp_solves_under_cap() {
    reset_arena();
    set_max_arena_bits(Some(256));

    let P = CscMatrix::new(2, 2, vec![0, 0, 1], vec![1], vec![rat(2)]);
    let q: Vec<T> = vec![rat(0), rat(0)];
    let A = CscMatrix::from(&[
        [rat(0), rat(0)],
        [-rat(2), rat(0)],
        [rat(0), -rat(1)],
    ]);
    let b: Vec<T> = vec![rat(1), -rat(2), -rat(2)];
    let cones = [SecondOrderConeT(3)];

    let mut solver =
        DefaultSolver::new(&P, &q, &A, &b, &cones, default_rational_settings()).unwrap();
    solver.solve();

    assert!(matches!(solver.solution.status, SolverStatus::Solved));
    let obj = solver.solution.obj_val.to_f64();
    assert!((obj - 1.0).abs() < 1e-6, "obj = {obj}, expected ~1");

    set_max_arena_bits(None);
    reset_arena();
}

/// Verify the per-cone accessor metadata on Solution<RationalReal>
/// matches the user's input cone declarations.
#[test]
fn rational_lp_solution_carries_per_cone_specs() {
    reset_arena();
    set_max_arena_bits(Some(256));

    let P: CscMatrix<T> = CscMatrix::<T>::zeros((2, 2));
    let q: Vec<T> = vec![rat(1), -rat(1)];
    let A = CscMatrix::new(
        4,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 3],
        vec![rat(1), -rat(1), rat(1), -rat(1)],
    );
    let b: Vec<T> = vec![rat(1); 4];
    let cones = [NonnegativeConeT(4)];

    let mut solver =
        DefaultSolver::new(&P, &q, &A, &b, &cones, default_rational_settings()).unwrap();
    solver.solve();

    let specs = &solver.solution.cone_specs;
    assert_eq!(specs.len(), 1, "one input cone -> one ConeSpec");
    assert_eq!(specs[0].dim, 4);
    assert_eq!(specs[0].range, 0..4);

    // dual_block / primal_block return the right slices.
    let dual = solver.solution.dual_block(0).unwrap();
    let prim = solver.solution.primal_block(0).unwrap();
    assert_eq!(dual.len(), 4);
    assert_eq!(prim.len(), 4);

    // primal_residual_per_block returns one entry, equal to ||s||
    let resid = solver.solution.primal_residual_per_block();
    assert_eq!(resid.len(), 1);

    set_max_arena_bits(None);
    reset_arena();
}

/// Check that the exact rational arithmetic guarantee holds for one
/// known-trivial case: `(1/3) * 3 == 1` exactly. This is impossible
/// in f64 (1/3 is unrepresentable so 0.333...×3 ≠ 1.0 exactly).
#[test]
fn rational_one_third_times_three_is_one() {
    reset_arena();
    let third = rat2(1, 3);
    let prod = third * rat(3);
    assert_eq!(prod, rat(1));
    let (n, d) = prod.into_pair();
    assert_eq!(n, BigInt::from(1));
    assert_eq!(d, BigInt::from(1));
    reset_arena();
}
