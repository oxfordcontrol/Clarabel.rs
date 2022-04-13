use num_traits::{Float, NumAssign, NumCast};

pub trait FloatT: 'static + Float + NumAssign + NumCast + std::iter::Sum + std::cmp::PartialOrd {}
impl FloatT for f32 {}
impl FloatT for f64 {}

// T = transpose, N = non-transposed
#[derive(PartialEq, Eq)]
pub enum MatrixShape {
    N,
    T,
}

pub trait VectorMathOps<T> {

    //scalar mut operations
    fn translate(&mut self, c: T);
    fn scale(&mut self, c: T);
    fn reciprocal(&mut self);

    //norms and friends
    fn dot(&self, y: &[T]) -> T;
    fn sumsq(&self) -> T;
    fn norm(&self) -> T;
    fn norm_inf(&self) -> T;
    fn norm_one(&self) -> T;

    //stats
    fn minimum(&self) -> T;
    fn maximum(&self) -> T;
    fn mean(&self) -> T;

    //blas-like vector ops
    fn axpby(&mut self, a: T, x: &[T], b :T);  //self = a*x+b*self
    fn waxpby(&mut self, a: T, x: &[T], b :T, y: &[T]); //self = a*x+b*y
}

pub mod native;
pub use native::*;
