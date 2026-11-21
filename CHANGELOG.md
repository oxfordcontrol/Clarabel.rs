# Changelog

Changes for the Rust version of Clarabel are documented in this file. For the Julia version, see [here](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Version numbering in this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  We aim to keep the core solver functionality and minor releases in sync between the Rust/Python and Julia implementations.  Small fixes that affect one implementation only may result in the patch release versions differing.

## Unreleased

### Rust-specific changes

- **Decoupled `FloatT` from `num_traits::Float`.** The solver internally now requires a smaller trait split — `Transcendental` (sqrt/ln/exp/powf/powi/recip plus sin/cos/atan2 for the analytic 3×3 eigensolver), `RealConst` (PI/SQRT_2/FRAC_1_SQRT_2), and `RealSentinel` (infinity/nan/epsilon/min/max/is_finite/is_sign_negative) — instead of the IEEE-only `num_traits::Float`/`FloatConst` bound. f32/f64 keep working through blanket impls; the change is invisible to `f64` users.
- **Relaxed `CoreFloatT: Copy` to `Clone`.** Enables non-Copy backends (e.g. arbitrary-precision rational, MPFR float) to satisfy `FloatT`. ~700 mechanical `.clone()` insertions across the algebra, KKT, and cone code paths. f64/f32 numerics are unaffected — `Clone::clone` on a `Copy` type is a register-move identical to `Copy`.
- **Added experimental exact-rational backend** behind the new `bigrational` cargo feature. Provides `RationalReal` — a `Copy`-able 4-byte arena handle whose arithmetic is backed by `num_rational::BigRational`. Bit-exact LP/QP iterates in default mode; opt-in inner-loop precision capping via `set_max_arena_bits(Some(p))` gives p-bit-bounded rounding for tractable runtime on non-trivial problems. Mutually exclusive with `sdp` and `faer-sparse` (those features pin `T` to f32/f64 for BLAS / `faer::RealField`). See `README.md` and `examples/rust/example_lp_rational.rs` for details.
- New `clarabel::algebra` re-exports for the rational backend: `RationalReal`, `arena_len`, `reset_arena`, `precision_bits`, `set_precision_bits`, `with_precision`, `max_arena_bits`, `set_max_arena_bits`, `with_max_arena_bits`.

## [0.11.1] - 2025-11-06

### Rust-specific changes

- Prefer static linking for BLAS and LAPACK libraries in Rust builds [#192](https://github.com/oxfordcontrol/Clarabel.rs/pull/192)


## [0.11.0] - 2025-21-05

### Changed

- Implemented LDL :auto select option [#162](https://github.com/oxfordcontrol/Clarabel.rs/pull/162)

- Consecutive 1D cones are collapsed and aggregated to a nonnegative cone [#163](https://github.com/oxfordcontrol/Clarabel.rs/pull/163)

- Added option to drop structural zeros during problem creation [#180](https://github.com/oxfordcontrol/Clarabel.rs/pull/180)


### Rust-specific changes

- Configurable print streams [#160](https://github.com/oxfordcontrol/Clarabel.rs/pull/160)

- Update to latest blas and lapack src crates by @bnaras in [#161](https://github.com/oxfordcontrol/Clarabel.rs/pull/161)

- Support for Panua and MKL Pardiso [#170](https://github.com/oxfordcontrol/Clarabel.rs/pull/170) [#175](https://github.com/oxfordcontrol/Clarabel.rs/pull/175) [#181](https://github.com/oxfordcontrol/Clarabel.rs/pull/181)

- Custom termination callback hooks for Rust, Python and C [#176](https://github.com/oxfordcontrol/Clarabel.rs/pull/176) [#182](https://github.com/oxfordcontrol/Clarabel.rs/pull/182) [#189](https://github.com/oxfordcontrol/Clarabel.rs/pull/189)

- Faster implementations for 2x2 and 3x3 matrix decompositions [#176](https://github.com/oxfordcontrol/Clarabel.rs/pull/176)

- Improved robustness of 2-norm computations [#184](https://github.com/oxfordcontrol/Clarabel.rs/pull/184)

- Added settings validation checks during problem creation and updating [#185](https://github.com/oxfordcontrol/Clarabel.rs/pull/176)

- Fixes for issues [#171](https://github.com/oxfordcontrol/Clarabel.rs/issues/171) and [#187](https://github.com/oxfordcontrol/Clarabel.rs/issues/187)






## [0.10.0] - 2025-03-02

### Changed
- fix socp line search failure case [#141](https://github.com/oxfordcontrol/Clarabel.rs/pull/141)
- norm unscaling bug fix [#136](https://github.com/oxfordcontrol/Clarabel.rs/pull/136)
- added `max_threads` to settings

### Rust-specific changes
- force mkl LP64 format (32 bit ints) [#130](https://github.com/oxfordcontrol/Clarabel.rs/pull/130)
- python 3.7/3.8 EOL updates [#147](https://github.com/oxfordcontrol/Clarabel.rs/pull/147)
- enable CSC diagonal counting for triu/tril [#145](https://github.com/oxfordcontrol/Clarabel.rs/pull/145)
- use .dlext directly from Libdl rather than Base.Libc in Clarabel.Rs julia wrapper by @mipals [#142](https://github.com/oxfordcontrol/Clarabel.rs/pull/142)
- wasm as platform dependency.  Fixes #135 [#139](https://github.com/oxfordcontrol/Clarabel.rs/pull/139)
- fix of #125 [#138](https://github.com/oxfordcontrol/Clarabel.rs/pull/138)
- fix #127 indexing failure in presolve [#135](https://github.com/oxfordcontrol/Clarabel.rs/pull/137)
- utilities for converting to CSC canonicalization / deduplication [#140](https://github.com/oxfordcontrol/Clarabel.rs/pull/140)
- fix compilation failure without "serde" feature by @cbbowen in [#131](https://github.com/oxfordcontrol/Clarabel.rs/pull/131)
- release gil when solving by @wuciting in [#122](https://github.com/oxfordcontrol/Clarabel.rs/pull/122)
- allow python to build with non-scipy blas and lapack [#151](https://github.com/oxfordcontrol/Clarabel.rs/pull/151)
- QP/SDP tests for pytest [#150](https://github.com/oxfordcontrol/Clarabel.rs/pull/150)
- update to faer v0.21 [#155](https://github.com/oxfordcontrol/Clarabel.rs/pull/155)
- python data updates [#156](https://github.com/oxfordcontrol/Clarabel.rs/pull/156)
- cvxpy support hooks for #75 in [#157](https://github.com/oxfordcontrol/Clarabel.rs/pull/157)
- Additional documentation 

## [0.9.0] - 2024-01-06

### Changed
- Read/write problems to JSON files [#111](https://github.com/oxfordcontrol/Clarabel.rs/pull/111)

### Rust specific changes
- validation tools for solver settings [#113](https://github.com/oxfordcontrol/Clarabel.rs/pull/113)

- adds feature to include supernodal LDL solver from `faer-rs` [#112](https://github.com/oxfordcontrol/Clarabel.rs/pull/112)

- Add wasm feature by @alexarice in [#114](https://github.com/oxfordcontrol/Clarabel.rs/pull/114)
- pypi and testpypi build updates by @tschm in [#110](https://github.com/oxfordcontrol/Clarabel.rs/pull/110), [#115](https://github.com/oxfordcontrol/Clarabel.rs/pull/115), [#109](https://github.com/oxfordcontrol/Clarabel.rs/pull/109)



## [0.8.1] - 2024-21-05
### Changed 

- change to docs.rs configuration so that SDP documentation will build

## [0.8.0] - 2024-21-05
### Changed 

- implements chordal decomposition for PSD cones [#100](https://github.com/oxfordcontrol/Clarabel.rs/pull/100)
- updates scaling bounds. Fixes [#96](https://github.com/oxfordcontrol/Clarabel.rs/issues/96)

### Rust specific changes

- Derive debug trait to the solution struct [#97](https://github.com/oxfordcontrol/Clarabel.rs/pull/97). Thanks @nunzioono.
- Resolve clippy warnings for rustc >=v1.75 [#94](https://github.com/oxfordcontrol/Clarabel.rs/pull/94)

## [0.7.1] - 2024-29-02
### Changed 

- Fixes a panic / crash condition in PSD scaling step [#78](https://github.com/oxfordcontrol/Clarabel.rs/pull/78)

### Rust specific changes

- Fix to output printing when Python version is run within a Jupyter notebook / Google Colab.  Fixes [#60].

## [0.7.1] - 2024-29-02
### Changed 

- Fixes a panic / crash condition in PSD scaling step [#78](https://github.com/oxfordcontrol/Clarabel.rs/pull/78)

### Rust specific changes

- Fix to output printing when Python version is run within a Jupyter notebook / Google Colab.  Fixes [#60](https://github.com/oxfordcontrol/Clarabel.rs/issues/60).


## [0.7.0] - 2024-26-02
### Changed 

- Solution output reports dual objective values.  Infeasible problems report NaN. Fixes [#67] (https://github.com/oxfordcontrol/Clarabel.rs/issues/67)
- Solver now supports problems with a mix of PSD and nonsymmetric cones. Fixes [#66] (https://github.com/oxfordcontrol/Clarabel.rs/issues/66)
- Added methods for updating problem data without reallocating memory.  Addresses [#59] (https://github.com/oxfordcontrol/Clarabel.rs/issues/59)
- Bug fix enforcing scaling limits in equilibration.  Port of Julia fix [#151](https://github.com/oxfordcontrol/Clarabel.jl/pull/151)
- Bug fix in infeasibility detection. Fixes [#65] (https://github.com/oxfordcontrol/Clarabel.rs/issues/65)



## [0.6.0] - 2023-20-09
### Changed 

This version introduces support for the generalized power cone and implements stability and speed improvements for SOC problems.  SOCs with
dimension less than or equal to 4 are now treated as special cases with dense Hessian blocks.

- Introduces support for the generalized power cone. 
- Implements stability and speed improvements for SOC problems.  SOCs with dimension less than or equal to 4 are now treated as special cases with dense Hessian blocks.
- Fixes bad initialization point for non-quadratic objectives 
- Improved convergence speed for QPs with no constraints or only ZeroCone constraints.
- Internal code restructuring for cones with sparsifiable Hessian blocks.

### Rust specific changes
- Added additional documentation and utilities [#43](https://github.com/oxfordcontrol/Clarabel.rs/issues/43),[#46](https://github.com/oxfordcontrol/Clarabel.rs/issues/46).
- Allow printing of internal timers through Julia wrappers in ClarabelRs [#44](https://github.com/oxfordcontrol/Clarabel.rs/issues/44)
- Updated keywords for crates.io [#45](https://github.com/oxfordcontrol/Clarabel.rs/issues/45)
- Better error reporting from internal QDLDL factor methods.  Fixes [#49](https://github.com/oxfordcontrol/Clarabel.rs/issues/49)


## [0.5.1] - 2023-02-06
### Changed 
Fixes convergence edge case in KKT direct solve iterative refinement.

### Rust specific changes
- Updates for custom build of R interface with BLAS/LAPACK support 
- Additional linear algebra unit tests and CSC matrix utilities


## [0.5.0] - 2023-25-04
### Changed 

This version ports support for PSD cones from the Julia version to Rust, with internal supporting modifications to both versions to keep implementations synchronized.

### Rust specific changes

- Package implements a variety of dense linear algebra functions using BLAS/LAPACK in support of the PSD cone implementation.   Provides various build options for using different BLAS/LAPACK external libraries.

- Python interface makes direct access to SciPy BLAS/LAPACK internal function pointers so that no external library linking is required for distribution via PyPI.


## [0.4.1] - 2023-08-03

### Changed 

Added optional feature to remove inequality constraints with very large upper bounds.   This feature is enabled by default but can be turned off using the `presolve_enable` setting.  

Bug fix in equilibration for NN and zero cones.
### Rust/Python specific changes

Rust algebra module modified to allow chaining of elementwise vector operations.

Added Rust matrix format checking utility in CscMatrix::check_format.  NB: CscMatrix 
integrity is assumed by the solver and is not checked internall.
## [0.4.0] - 2023-25-02

### Changed 

- Internal fixes relating to initialization of iterates in symmetric cone problems.

- Numerical stability improvements for second order cone constraints. 

### Rust/Python specific changes

- Modification of the internal calls to the Rust qdldl to allow for direct assignment of parameters in AMD ordering.   

- Added release of binaries for arm64 Linux [#9](https://github.com/oxfordcontrol/Clarabel.rs/issues/9).   Thanks to @nrontsis.

- Fixed a bug using `==` for SolverStatus objects in Python.  Fixes [#10](https://github.com/oxfordcontrol/Clarabel.rs/issues/10).

- Python now reports the Clarabel version using `__version__` instead of `__version__()`.

- Added additional unit tests for Rust implementation.   NB: Rust implementation is also tested
offline against the Julia-based benchmark problem suite, but this will not appear in coverage reporting.  


## [0.3.0] - 2022-09-13

### Changed 

- Implements support for exponential and power cones

- Numerical stability improvements

- Various bug fixes

## [0.2.0] - 2022-07-31

- Rust/python implementation released starting from this version.

- Ported all documentation to the common site [here](https://github.com/oxfordcontrol/ClarabelDocs)

[0.11.1]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.10.0...v0.11.1
[0.11.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/oxfordcontrol/Clarabel.rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/oxfordcontrol/Clarabel.rs/tree/v0.2.0
