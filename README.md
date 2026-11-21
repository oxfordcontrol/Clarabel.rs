<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-dark-rs.png">
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-light-rs.png">
  <img alt="Clarabel.jl logo" src="https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/logo-banner-light-rs.png" width="66%">
</picture>
<h1 align="center" margin=0px>
Interior Point Conic Optimization for Rust and Python
</h1>
<p align="center">
   <a href="https://github.com/oxfordcontrol/Clarabel.rs/actions"><img src="https://github.com/oxfordcontrol/Clarabel.rs/workflows/ci/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/oxfordcontrol/Clarabel.rs"><img src="https://codecov.io/gh/oxfordcontrol/Clarabel.rs/branch/main/graph/badge.svg"></a>
  <a href="https://clarabel.org"><img src="https://img.shields.io/badge/Documentation-stable-purple.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/oxfordcontrol/Clarabel.rs/releases"><img src="https://img.shields.io/badge/Release-v0.11.1-blue.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://clarabel.org">Documentation</a>
</p>

__Clarabel.rs__ is a Rust implementation of an interior point numerical solver for convex optimization problems using a novel homogeneous embedding.  Clarabel.rs solves the following problem:

$$
\begin{array}{r}
\text{minimize} & \frac{1}{2}x^T P x + q^T x\\\\[2ex]
 \text{subject to} & Ax + s = b \\\\[1ex]
        & s \in \mathcal{K}
 \end{array}
$$

with decision variables
$x \in \mathbb{R}^n$,
$s \in \mathbb{R}^m$
and data matrices
$P=P^\top \succeq 0$,
$q \in \mathbb{R}^n$,
$A \in \mathbb{R}^{m \times n}$, and
$b \in \mathbb{R}^m$.
The convex set $\mathcal{K}$ is a composition of convex cones.

__For more information see the Clarabel Documentation ([stable](https://clarabel.org) |  [dev](https://clarabel.org/dev)).__

Clarabel is also available in a Julia implementation.  See [here](https://github.com/oxfordcontrol/Clarabel.jl).
 

## Features

* __Versatile__: Clarabel.rs solves linear programs (LPs), quadratic programs (QPs), second-order cone programs (SOCPs) and semidefinite programs (SDPs). It also solves problems with exponential, power cone and generalized power cone constraints.
* __Quadratic objectives__: Unlike interior point solvers based on the standard homogeneous self-dual embedding (HSDE), Clarabel.rs handles quadratic objectives without requiring any epigraphical reformulation of the objective.   It can therefore be significantly faster than other HSDE-based solvers for problems with quadratic objective functions.
* __Infeasibility detection__: Infeasible problems are detected using a homogeneous embedding technique.
* __Open Source__: Our code is available on [GitHub](https://github.com/oxfordcontrol/Clarabel.rs) and distributed under the Apache 2.0 License

## Exact-arithmetic backend (experimental)

The optional `bigrational` Cargo feature replaces the default `f64` scalar
type with `RationalReal` — an arbitrary-precision rational backed by
[`num_rational::BigRational`](https://docs.rs/num-rational) stored in a
thread-local arena. LP/QP iterates are bit-exact rationals; SOCP/exp/pow
barrier transcendentals (`sqrt`, `ln`, `exp`, `powf`) are computed at a
configurable thread-local working precision via Newton/Taylor iterations.

> ⚠️ **`bigrational` does not support SDPs.** PSD cone projection requires
> eigendecomposition, which is not exact in rationals. The feature
> emits a `compile_error!` if combined with `sdp`/`sdp-*` or
> `faer-sparse` (those features pin `T` to `f32`/`f64` for BLAS / faer's
> `RealField`). For high-precision SDP use the `mpfr` backend below
> with the existing BLAS path; for **certified rational SDP solutions**,
> solve in `f64`/`MpfrFloat` and round the dual back to rationals via
> a Peyrl–Parrilo-style tightening step using the provided
> `tighten_scalar`, `tighten_vec`, and `tighten_psd_block` helpers in the
> `clarabel::algebra::rational` module.

```toml
[dependencies]
clarabel = { version = "0", default-features = false, features = ["serde", "bigrational"] }
```

```rust
use clarabel::algebra::*;
use clarabel::solver::*;

let mut solver = DefaultSolver::<RationalReal>::new(&P, &q, &A, &b, &cones, settings)?;
solver.solve();

// Extract the primal solution as exact rationals.
for xi in &solver.solution.x {
    let (numer, denom) = xi.into_pair();   // BigInt, BigInt
    println!("{} / {}", numer, denom);
}
```

Two precision modes:

- **Exact** (default): `+`, `-`, `*`, `/` on `RationalReal` are exact and
  unbounded. The headline guarantee `(1/3) + (1/3) + (1/3) == 1` holds
  exactly. Per-iteration BigRational denominators grow geometrically so
  each subsequent operation gets slower; this is intrinsic to exact
  rational LP solving and limits practical use to small problems.
- **Bounded-precision**: call `set_max_arena_bits(Some(p))` to round
  arithmetic results to `m / 2ᵖ` whenever a numerator or denominator
  would otherwise exceed `p` bits. Recommended values: `Some(256)` for
  general-purpose runs (~77 decimal digits, ~5× the precision of `f64`),
  `Some(167)` for ≥ 50 decimal digits.

The end-to-end `examples/rust/example_lp_rational.rs` solves a 2-D box
LP in ~50 ms with `set_max_arena_bits(Some(256))` (10663 arena entries,
6 IPM iterations bit-identical to the f64 baseline, primal `x` returned
as exact 256-bit-bounded rationals). Without the cap the same example
takes minutes per iteration as denominators grow geometrically.

Tolerance semantics in exact mode:

- The solver's stopping criteria (`tol_feas`, `tol_gap_abs`, `tol_gap_rel`) are typed `T` and compared against `T`-typed residuals. With `T = RationalReal` the comparisons are *exact* — `r_prim < tol_feas` is true iff the rational `r_prim` is strictly less than the rational `tol_feas`.
- The `1e-8` defaults that `DefaultSettings::default()` would copy from the f64 path are constructed via `T::from_f64(1e-8)` and so capture the **exact IEEE-binary** rational `5764607523034235/2^79`, not the decimal `1/100_000_000`. To set tolerances by intent rather than by IEEE round-trip, build settings explicitly:
  ```rust
  let tol = RationalReal::from_pair(BigInt::from(1), BigInt::from(100_000_000));
  let settings = DefaultSettingsBuilder::<RationalReal>::default()
      .tol_feas(tol).tol_gap_abs(tol).tol_gap_rel(tol)
      .build()?;
  ```
- A `tol = T::zero()` setting is permitted and means "terminate iff the residuals are exactly zero" — only achievable for problems whose optima happen to be rational and reachable by a finite IPM trajectory. For most problems use a small positive rational; the headline exactness guarantee is on the *iterates*, not on the termination test.
- When `set_max_arena_bits(Some(p))` is engaged, residual comparisons see the rounded values, so `tol = T::zero()` will almost never fire and a positive rational `tol` is required as in f64 mode.
- `Solution<RationalReal>` derives `serde::Serialize`/`Deserialize` (with `T: Serialize + DeserializeOwned`), so witness JSON files preserve the exact `(numer, denom)` pair across save/load — useful for QOU-style nuclear-mass certificates.

Limitations and feature interactions:

- Mutually exclusive with `sdp` and `faer-sparse` — those features pin
  `T` to `f32`/`f64` for BLAS/LAPACK and `faer::RealField` operations.
- The `Send + Sync` claim on `RationalReal` is upheld by an
  `unsafe impl` with the documented invariant that a value may only be
  dereferenced on the thread that produced it. Within `solver.solve()`
  this is satisfied (Clarabel's IPM is single-threaded for a given
  problem). Outer parallelism — independent solves on independent
  threads, each with its own thread-local arena — is supported.
- Memory grows monotonically during a solve. Call
  `clarabel::algebra::reset_arena()` between solves to recover.

See `examples/rust/example_lp_rational.rs` for an end-to-end demo.

# Installation

Clarabel can be imported to Cargo based Rust projects by adding
```rust
[dependencies]
clarabel = "0"  
```
to the project's `Cargo.toml` file.   To install from source, see the [Rust Installation Documentation](https://oxfordcontrol.github.io/ClarabelDocs/stable/rust/installation_rs/).

To use the Python interface to the solver:
```
pip install clarabel
```

To install the Python interface from source, see the [Python Installation Documentation](https://oxfordcontrol.github.io/ClarabelDocs/stable/python/installation_py/).

## Citing
```
@misc{Clarabel_2024,
      title={Clarabel: An interior-point solver for conic programs with quadratic objectives}, 
      author={Paul J. Goulart and Yuwen Chen},
      year={2024},
      eprint={2405.12762},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## License 🔍
This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details.
