#![allow(non_snake_case)]
use num_traits::{FromPrimitive, Num, NumAssign, Signed};
use std::fmt::{Debug, Display, LowerExp};

#[cfg(feature = "sdp")]
use crate::algebra::dense::BlasFloatT;

use super::transcendental::{BitWidthDiagnostic, RealConst, RealSentinel, Transcendental};

// Phase 2 of the BigRational backend (deferred):
// `CoreFloatT` currently still requires `Copy`. The originally-proposed
// `Rc<BigRational>`-with-Copy-semantics newtype is not implementable in
// safe Rust (`Rc<T>` has a `Drop` impl for the refcount and so cannot
// itself be `Copy`).  The selected resolution is to relax `Copy` to
// `Clone` and add explicit `.clone()` insertions at by-value call
// sites — a roughly-1000-error mechanical refactor concentrated in the
// cone files (powcone/expcone/socone/genpowcone). It is deferred to a
// follow-up commit so that the trait-split Phase 1 can land for review
// on its own and the f64 baseline stays bisectable. The rational
// backend will use `Arc<BigRational>` since `CoreFloatT: Send + Sync`.
//
// `vecmath.rs` already contains `.clone()`-style call patterns ahead of
// the trait change; on `Copy` types the clones compile to the same
// register/memcpy moves as before, so f64 numerics are unaffected.

/// Core traits for internal floating point values.
///
/// This trait defines a subset of bounds for `FloatT`, which is preferred
/// throughout for use in the solver.  When the `sdp` feature is enabled,
/// `FloatT` is additionally restricted to f32/f64 types supported by BLAS.
/// When the `faer-sparse` feature is enabled, `FloatT` is additionally
/// restricted to types implementing `RealField` from the `faer` crate.
///
/// Note: `Float`/`FloatConst` are not required directly. Instead the solver
/// requires the smaller [`Transcendental`], [`RealConst`] and [`RealSentinel`]
/// trait split, which is satisfied via blanket impls for any IEEE float
/// (`f32`, `f64`) and is also implementable for non-IEEE backends such as
/// `RationalReal` (exact rational) and a future MPFR backend.
pub trait CoreFloatT:
    'static
    + Send
    + Sync
    + Num
    + NumAssign
    + Signed
    + Clone
    + PartialOrd
    + Default
    + FromPrimitive
    + Display
    + LowerExp
    + Debug
    + Sized
    + Transcendental
    + RealConst
    + RealSentinel
    + BitWidthDiagnostic
{
}

impl<T> CoreFloatT for T where
    T: 'static
        + Send
        + Sync
        + Num
        + NumAssign
        + Signed
        + Clone
        + PartialOrd
        + Default
        + FromPrimitive
        + Display
        + LowerExp
        + Debug
        + Sized
        + Transcendental
        + RealConst
        + RealSentinel
        + BitWidthDiagnostic
{
}

// additional traits that add bounds to CoreFloatT
// when optional dependencies are enabled

// if "sdp" is enabled, we must add an additional trait
// trait bound to restrict compilation for f32/f64 types
// since there is no BLAS support otherwise

cfg_if::cfg_if! {
    if #[cfg(feature="sdp")] {
        /// A marker trait implemented on types supported by BLAS (i.e. f32 and f64)
        /// when the package is compiled with the "sdp" feature using a BLAS/LAPACK library.
        ///
        /// Pulls in `Copy + NumCast + ToPrimitive`:
        /// - `Copy` because the SDP code paths (chordal decomposition,
        ///   dense fixed-size 3x3 eigen/svd, BLAS Cholesky) were written
        ///   against the original IEEE-only `FloatT: Float` bound and
        ///   contain hundreds of by-value reads from `Vec<T>` indexes.
        ///   Adding `Copy` here restores those at exactly the right
        ///   scope (the trait is impl'd only for f32/f64, both of which
        ///   are `Copy`) without forcing the cones / algebra layer back
        ///   to a global `Copy` requirement.
        /// - `NumCast`/`ToPrimitive` because the BLAS code paths use
        ///   `T::from(usize)` and `value.to_i32()` (e.g. for LAPACK
        ///   workspace sizing and SVD truncation tolerances).
        ///
        /// All three bounds are vacuous for the only impl types
        /// (`f32`, `f64`).
        #[doc(hidden)]
        pub trait MaybeBlasFloatT
            : BlasFloatT + Copy + num_traits::NumCast + num_traits::ToPrimitive {}
        impl<T> MaybeBlasFloatT for T
            where T: BlasFloatT + Copy + num_traits::NumCast + num_traits::ToPrimitive {}
    }
    else {
        #[doc(hidden)]
        pub trait MaybeBlasFloatT {}
        impl<T> MaybeBlasFloatT for T {}
    }
}

// if "faer" is enabled, we must add an additional bound
// to restrict compilation to types implementing RealField

cfg_if::cfg_if! {
    if #[cfg(feature="faer-sparse")] {
        #[doc(hidden)]
        /// A marker trait implemented on types supported by faer-rs
        /// when the package is compiled with the "faer-sparse" feature
        pub trait MaybeFaerFloatT : faer_traits::RealField {}
        impl<T> MaybeFaerFloatT for T where T: faer_traits::RealField {}
    }
    else {
        #[doc(hidden)]
        pub trait MaybeFaerFloatT {}
        impl<T> MaybeFaerFloatT for T {}
    }
}

/// Main trait for floating point types used in the Clarabel solver.
///
/// All floating point calculations in Clarabel are represented internally on values
/// implementing the `FloatT` trait, with implementations provided only for f32 and f64
/// native types when compiled with BLAS/LAPACK support for SDPs. If SDP support is not
/// enabled then it should be possible to compile Clarabel to support any any other
/// floating point type provided that it satisfies the trait bounds of `CoreFloatT`.
///
/// `FloatT` is built from a small split of supertraits ([`Transcendental`],
/// [`RealConst`], [`RealSentinel`]) plus the standard `num_traits` arithmetic
/// hierarchy, rather than `num_traits::Float`, so non-IEEE backends (e.g.
/// the `bigrational` feature) can also satisfy it.
pub trait FloatT: CoreFloatT + MaybeBlasFloatT + MaybeFaerFloatT {}
impl<T> FloatT for T where T: CoreFloatT + MaybeBlasFloatT + MaybeFaerFloatT {}

/// Trait for converting Rust primitives to [`FloatT`](crate::algebra::FloatT)
///
/// This convenience trait is implemented on f32/64 and u32/64.  This trait
/// is required internally by the solver for converting constant primitives
/// to [`FloatT`](crate::algebra::FloatT).  It is also used by the
/// [user settings](crate::solver::implementations::default::DefaultSettings)
/// for converting defaults of primitive type to [`FloatT`](crate::algebra::FloatT).
//
// NB: `AsFloatT` is a convenience trait for f32/64 and u32/64
// so that we can do things like (2.0).as_T() everywhere on
// constants, rather than the awful T::from_f32(2.0).unwrap()
pub(crate) trait AsFloatT<T>: 'static {
    fn as_T(&self) -> T;
}

macro_rules! impl_as_FloatT {
    ($ty:ty, $ident:ident) => {
        impl<T> AsFloatT<T> for $ty
        where
            T: std::ops::Mul<T, Output = T> + FromPrimitive + 'static,
        {
            #[inline]
            fn as_T(&self) -> T {
                T::$ident(*self).unwrap()
            }
        }
    };
}
impl_as_FloatT!(u32, from_u32);
impl_as_FloatT!(u64, from_u64);
impl_as_FloatT!(usize, from_usize);
impl_as_FloatT!(f32, from_f32);
impl_as_FloatT!(f64, from_f64);
