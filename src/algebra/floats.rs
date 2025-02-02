#![allow(non_snake_case)]
use num_traits::{Float, FloatConst, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display, LowerExp};

#[cfg(feature = "sdp")]
use crate::algebra::dense::BlasFloatT;

/// Core traits for internal floating point values.
///
/// This trait defines a subset of bounds for `FloatT`, which is preferred
/// throughout for use in the solver.  When the `sdp` feature is enabled,
/// `FloatT` is additionally restricted to f32/f64 types supported by BLAS.
/// When the `faer-sparse` feature is enabled, `FloatT` is additionally
/// restricted to types implementing `RealField` from the `faer` crate.
pub trait CoreFloatT:
    'static
    + Send
    + Sync
    + Float
    + FloatConst
    + NumAssign
    + Default
    + FromPrimitive
    + Display
    + LowerExp
    + Debug
    + Sized
{
}

impl<T> CoreFloatT for T where
    T: 'static
        + Send
        + Sync
        + Float
        + FloatConst
        + NumAssign
        + Default
        + FromPrimitive
        + Display
        + LowerExp
        + Debug
        + Sized
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
        /// when the package is compiled with the "sdp" feature using a BLAS/LAPACK library
        #[doc(hidden)]
        pub trait MaybeBlasFloatT : BlasFloatT {}
        impl<T> MaybeBlasFloatT for T where T: BlasFloatT {}
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
/// `FloatT` relies on [`num_traits`](num_traits) for most of its constituent trait bounds.
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
