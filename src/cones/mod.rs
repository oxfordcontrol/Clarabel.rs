//PJG: Are both use and mod required here?

#![allow(dead_code)]        //PJG: temporary.  Remove
#![allow(unused_variables)] //PJG: temporary.  Remove

pub mod nonnegativecone;
pub mod zerocone;
pub mod socone;
pub mod coneset;
pub use nonnegativecone::*;
pub use zerocone::*;
pub use socone::*;
pub use coneset::*;

use crate::algebra::*;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum SupportedCones {
    ZeroConeT,
    NonnegativeConeT,
    SecondOrderConeT,
}

//PJG: translation of Julia in this function is
//probably not the best way, plus it's not a dict now
pub fn cone_dict<T: 'static + FloatT>(cone: SupportedCones, dim: usize) -> Box<dyn ConvexCone<T>> {
    match cone {
        SupportedCones::ZeroConeT => Box::new(ZeroCone::<T>::new(dim)),
        SupportedCones::NonnegativeConeT => Box::new(NonnegativeCone::<T>::new(dim)),
        SupportedCones::SecondOrderConeT => Box::new(SecondOrderCone::<T>::new(dim))
    }
}

pub trait ConvexCone<T: FloatT> {

    fn dim(&self) -> usize;
    fn degree(&self) -> usize;
    fn numel(&self) -> usize;
    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool;
    fn update_scaling(&mut self, s: &[T], z: &[T]);
    fn set_identity_scaling(&mut self);
    #[allow(non_snake_case)]
    fn get_WtW_block(&self, WtWblock: &mut [T]);
    fn λ_circ_λ(&self, x: &mut [T]);
    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]);
    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]);
    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]);
    fn shift_to_cone(&self, z: &mut [T]);
    #[allow(non_snake_case)]
    fn gemv_W(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T);
    #[allow(non_snake_case)]
    fn gemv_Winv(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T);
    fn add_scaled_e(&self, x: &mut [T], α: T);
    fn step_length(&self, dz: &[T], ds: &[T], z: &[T], s: &[T]) -> (T, T);
}
