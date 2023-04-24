mod core;
pub use self::core::*;
mod block_concatenate;
pub use block_concatenate::*;
mod matrix_math;
pub use matrix_math::*;

mod blaslike_traits;
pub(crate) use blaslike_traits::*;
mod blas;
pub(crate) use self::blas::*;
mod cholesky;
pub(crate) use cholesky::*;
mod syevr;
pub(crate) use syevr::*;
mod svd;
pub(crate) use svd::*;

mod gemm;
mod gemv;
mod kron;
mod symv;
mod syr2k;
mod syrk;
