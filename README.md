
<h1 align="center" margin=0px>
  <img src="https://github.com/oxfordcontrol/Clarabel.rs/blob/main//assets/logo-banner-light.png#gh-light-mode-only" width=60%>
  <img src="https://github.com/oxfordcontrol/Clarabel.rs/blob/main//assets/logo-banner-dark.png#gh-dark-mode-only"   width=60%>
  <br>
Interior Point Conic Optimization for Rust
</h1>
<p align="center">
   <a href="https://github.com/oxfordcontrol/Clarabel.rs/actions"><img src="https://github.com/oxfordcontrol/Clarabel.rs/workflows/ci/badge.svg?branch=main"></a>
  <a href="https://codecov.io/gh/oxfordcontrol/Clarabel.rs"><img src="https://codecov.io/gh/oxfordcontrol/Clarabel.rs/branch/master/graph/badge.svg"></a>
  <a href="https://oxfordcontrol.github.io/Clarabel.rs/stable"><img src="https://img.shields.io/badge/Documentation-stable-purple.svg"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://github.com/oxfordcontrol/Clarabel.rs/releases"><img src="https://img.shields.io/badge/Release-v0.1.1-blue.svg"></a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#license-">License</a> •
  <a href="https://oxfordcontrol.github.io/Clarabel.rss/stable">Documentation</a>
</p>

__Clarabel.rs__ is a Rust implementation of an interior point numerical solver for convex optimization problems using a novel homogeneous embedding.  Clarabel.rs solves the following problem:

<p align="center">
  <img src="https://github.com/oxfordcontrol/Clarabel.rs/blob/main/assets/problem_format-light.png#gh-light-mode-only" width=30%>
  <img src="https://github.com/oxfordcontrol/Clarabel.rs/blob/main/assets/problem_format-dark.png#gh-dark-mode-only"   width=30%>
</p>

with decision variables 
$x \in \mathbb{R}^n$,
$s \in \mathbb{R}^m$
and data matrices 
$P=P^\top \succeq 0$,
$q \in \mathbb{R}^n$, 
$A \in \mathbb{R}^{m \times n}$, and
$b \in \mathbb{R}^m$.
The convex set $\mathcal{K}$ is a composition of convex cones.


__For more information see the Clarabel.jl Documentation ([stable](https://oxfordcontrol.github.io/Clarabel.rs/stable) |  [dev](https://oxfordcontrol.github.io/Clarabel.rs/dev)).__

## Features

* __Versatile__: Clarabel.rss solves linear programs (LPs), quadratic programs (QPs) and second-order cone programs (SOCPs).  Future versions will provide support for problems involving positive semidefinite, exponential and power cones.
* __Quadratic objectives__: Unlike interior point solvers based on the standard homogeneous self-dual embedding (HSDE), Clarabel.rs handles quadratic objectives without requiring any epigraphical reformulation of the objective.   It can therefore be significantly faster than other HSDE-based solvers for problems with quadratic objective functions.
* __Infeasibility detection__: Infeasible problems are detected using a homogeneous embedding technique.
* __Open Source__: Our code is available on [GitHub](https://github.com/oxfordcontrol/Clarabel.rs) and distributed under the Apache 2.0 License

## License 🔍
This project is licensed under the Apache License - see the [LICENSE.md](LICENSE.md) file for details.
s