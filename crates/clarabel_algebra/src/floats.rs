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
{}
impl FloatT for f32 {}
impl FloatT for f64 {}

// AsFloatT is a convenience trait implemented on f32 / u32
// so that we can do things like (2.0).as_T() everywhere on 
// constants, rather than the awful T::from_f32(2.0).unwrap() 

#[allow(non_snake_case)]
pub trait AsFloatT<T> : 'static {
    fn as_T(&self)->T;
}

impl<T> AsFloatT<T> for f32 where T: FromPrimitive + 'static
{
    fn as_T(&self)->T { T::from_f32(*self).unwrap()}
}
impl<T> AsFloatT<T> for u32 where T: FromPrimitive + 'static
{
    fn as_T(&self)->T { T::from_u32(*self).unwrap()}
}

