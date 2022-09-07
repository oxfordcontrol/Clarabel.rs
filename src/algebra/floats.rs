use num_traits::{Float, FloatConst, FromPrimitive, NumAssign};

/// Trait for floating point types used in the Clarabel solver
///
/// All floating point calculations in Clarabel are represented internally on values
/// implementing the FloatT trait, with implementations provided for f32 and f64
/// native types. It should be possible to compile Clarabel to support any any other
/// floating point type provided that it satisfies the trait bounds of
/// [FloatT](crate::algebra::FloatT).
///
/// FloatT relies on [num_traits](num_traits) for most of its constituent trait bounds.

pub trait FloatT:
    'static
    + Send
    + Float
    + FloatConst
    + NumAssign
    + Default
    + FromPrimitive
    + std::fmt::Display
    + std::fmt::LowerExp
    + std::fmt::Debug
{
}
impl FloatT for f32 {}
impl FloatT for f64 {}

/// Trait for convering Rust primitives to [FloatT](crate::algebra::FloatT)
///
/// This convenience trait implemented on f32/64 and u32/64.  This trait
/// is required internally by the solver for converting constant primitives
/// to [FloatT](crate::algebra::FloatT).  It is also used by the
/// [user settings](crate::solver::implementations::default::DefaultSettings)
/// for converting defaults of primitive type to [FloatT](crate::algebra::FloatT).

// NB: AsFloatT is a convenience trait for f32/64 and u32/64
// so that we can do things like (2.0).as_T() everywhere on
// constants, rather than the awful T::from_f32(2.0).unwrap()

#[allow(non_snake_case)]
pub trait AsFloatT<T>: 'static {
    fn as_T(&self) -> T;
}

macro_rules! impl_as_T {
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
impl_as_T!(u32, from_u32);
impl_as_T!(u64, from_u64);
impl_as_T!(usize, from_usize);
impl_as_T!(f32, from_f32);
impl_as_T!(f64, from_f64);
