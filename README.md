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
