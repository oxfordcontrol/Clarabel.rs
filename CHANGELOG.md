# Changelog

Changes for the Rust version of Clarabel are documented in this file. For the Julia version, see [here](https://github.com/oxfordcontrol/Clarabel.jl/blob/main/CHANGELOG.md).

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Version numbering in this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).  We aim to keep the core solver functionality and minor releases in sync between the Rust/Python and Julia implementations.  Small fixes that affect one implementation only may result in the patch release versions differing.

## [0.9.0] - 2024-01-06

- ## What's Changed
- Read/write problems to JSON files [#111](https://github.com/oxfordcontrol/Clarabel.rs/pull/111)

## Rust specific changes
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
