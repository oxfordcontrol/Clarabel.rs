// Add a stylish Clarabel cow and gear logo to the docs
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/cow-and-gear-logo.png",
    html_favicon_url = "https://raw.githubusercontent.com/oxfordcontrol/ClarabelDocs/main/docs/src/assets/favicon.ico"
)]

//! <p align="center">
//! <picture>
//! <source srcset="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-dark-rs.png?raw=true" media="(prefers-color-scheme: dark)" width=50% >
//! <img src="https://github.com/oxfordcontrol/ClarabelDocs/blob/main/docs/src/assets/logo-banner-light-rs.png?raw=true" width=50% >
//! </picture>
//! </p>
//!
//!  __Clarabel.rs__ is a Rust implementation of an interior point numerical solver for convex optimization problems using a novel homogeneous embedding.  Clarabel solves the following problem:
//!
//! $$
//! \begin{array}{rl}
//! \text{minimize} & \frac{1}{2}x^T P x + q^T x\\\\\[2ex\]
//!  \text{subject to} & Ax + s = b \\\\\[1ex\]
//!         & s \in \mathcal{K}
//!  \end{array}
//! $$
//!
//!
//! with decision variables
//! $x \in \mathbb{R}^n$,
//! $s \in \mathbb{R}^m$
//! and data matrices
//! $P=P^\top \succeq 0$,
//! $q \in \mathbb{R}^n$,
//! $A \in \mathbb{R}^{m \times n}$, and
//! $b \in \mathbb{R}^m$.
//! The convex set $\mathcal{K}$ is a composition of convex cones.
//!
//! __For installation, tutorials and examples see the Clarabel User's Guide ([stable](https://oxfordcontrol.github.io/ClarabelDocs/stable) |  [dev](https://oxfordcontrol.github.io/ClarabelDocs/dev)).__
//!
//! Clarabel is also available in a Julia implementation.  See [here](https://github.com/oxfordcontrol/Clarabel.jl).
//!
//! ## Features
//!
//! * __Versatile__: Clarabel.rs solves linear programs (LPs), quadratic programs (QPs), second-order cone programs (SOCPs) and semidefinite programs (SDPs). It also solves problems with exponential, power cone and generalized power cone constraints.
//!
//! * __Quadratic objectives__: Unlike interior point solvers based on the standard homogeneous self-dual embedding (HSDE), Clarabel.rs handles quadratic objectives without requiring any epigraphical reformulation of the objective.   It can therefore be significantly faster than other HSDE-based solvers for problems with quadratic objective functions.
//!
//! * __Infeasibility detection__: Infeasible problems are detected using a homogeneous embedding technique.
//!
//! # Python interface
//!
//! Clarabel.rs comes with an optional Python interface.  See the [Python Installation Documentation](https://oxfordcontrol.github.io/ClarabelDocs/stable/python/installation_py/).
//!
//!
//! # License
//!
//! Licensed under Apache License, Version 2.0.  [LICENSE](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/LICENSE.md)

//Rust hates greek characters
#![allow(confusable_idents)]
#![warn(missing_docs)]

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod algebra;
pub mod qdldl;
pub mod solver;
pub(crate) mod stdio;
pub mod timers;

pub(crate) mod utils;
pub use crate::utils::infbounds::*;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "julia")]
pub mod julia;

#[allow(unused_macros)]
macro_rules! printbuildenv {
    ($tag:expr) => {
        if let Some(opt) = option_env!(concat!("VERGEN_", $tag)) {
            writeln!(crate::stdio::stdout(), "{}: {}", $tag, opt).unwrap();
        }
    };
}

/// print detailed build configuration info to stdout
#[allow(clippy::explicit_write)]
pub fn buildinfo() {
    use std::io::Write;

    #[cfg(feature = "buildinfo")]
    {
        printbuildenv!("BUILD_TIMESTAMP");
        printbuildenv!("CARGO_DEBUG");
        printbuildenv!("CARGO_FEATURES");
        printbuildenv!("CARGO_OPT_LEVEL");
        printbuildenv!("CARGO_TARGET_TRIPLE");
        printbuildenv!("RUSTC_CHANNEL");
        printbuildenv!("RUSTC_COMMIT_DATE");
        printbuildenv!("RUSTC_COMMIT_HASH");
        printbuildenv!("RUSTC_HOST_TRIPLE");
        printbuildenv!("RUSTC_LLVM_VERSION");
        printbuildenv!("RUSTC_SEMVER");
        printbuildenv!("SYSINFO_NAME");
        printbuildenv!("SYSINFO_OS_VERSION");
        printbuildenv!("SYSINFO_TOTAL_MEMORY");
        printbuildenv!("SYSINFO_CPU_VENDOR");
        printbuildenv!("SYSINFO_CPU_CORE_COUNT");
        printbuildenv!("SYSINFO_CPU_BRAND");
        printbuildenv!("SYSINFO_CPU_FREQUENCY");
    }
    #[cfg(not(feature = "buildinfo"))]
    writeln!(crate::stdio::stdout(), "no build info available").unwrap();
}

pub(crate) const _INFINITY_DEFAULT: f64 = 1e20;

#[test]
fn test_buildinfo() {
    buildinfo();
}
