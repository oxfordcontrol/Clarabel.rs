use num_traits::{Float, FromPrimitive, NumAssign};

// We use FloatT everywhere to allow for f32/f64 solvers
// or some future arbitrary precision implementation

pub trait FloatT:
    'static
    + Send
    + Float
    + NumAssign
    + Default
    + FromPrimitive
    + std::fmt::Display
    + std::fmt::LowerExp
{
}
impl FloatT for f32 {}
impl FloatT for f64 {}

// AsFloatT is a convenience trait for f32/64 and u32/64
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
            T: FromPrimitive + 'static,
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
impl_as_T!(f32, from_f32);
impl_as_T!(f64, from_f64);
