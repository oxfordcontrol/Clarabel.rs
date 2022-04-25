#![allow(non_snake_case)]
pub mod coneset;
pub mod nonnegativecone;
pub mod socone;
pub mod zerocone;
pub use coneset::*;
pub use nonnegativecone::*;
pub use socone::*;
pub use zerocone::*;

use crate::algebra::*;

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub enum SupportedCones {
    ZeroConeT,
    NonnegativeConeT,
    SecondOrderConeT,
}

//PJG: translation of Julia in this function is
//probably not the best way, plus it's not a dict now
pub fn cone_dict<T>(cone: SupportedCones, dim: usize) -> Box<dyn Cone<T, [T], [T]>>
where
    T: FloatT + 'static,
{
    match cone {
        SupportedCones::ZeroConeT => Box::new(ZeroCone::<T>::new(dim)),
        SupportedCones::NonnegativeConeT => Box::new(NonnegativeCone::<T>::new(dim)),
        SupportedCones::SecondOrderConeT => Box::new(SecondOrderCone::<T>::new(dim)),
    }
}

pub trait Cone<T, CV: ?Sized, BV: ?Sized> {
    fn dim(&self) -> usize;
    fn degree(&self) -> usize;
    fn numel(&self) -> usize;
    fn rectify_equilibration(&self, δ: &mut CV, e: &CV) -> bool;
    fn WtW_is_diagonal(&self) -> bool;
    fn update_scaling(&mut self, s: &CV, z: &CV);
    fn set_identity_scaling(&mut self);
    fn λ_circ_λ(&self, x: &mut CV);
    fn circ_op(&self, x: &mut CV, y: &CV, z: &CV);
    fn λ_inv_circ_op(&self, x: &mut CV, z: &CV);
    fn inv_circ_op(&self, x: &mut CV, y: &CV, z: &CV);
    fn shift_to_cone(&self, z: &mut CV);
    fn get_WtW_block(&self, WtWblock: &mut BV);
    fn gemv_W(&self, is_transpose: MatrixShape, x: &CV, y: &mut CV, α: T, β: T);
    fn gemv_Winv(&self, is_transpose: MatrixShape, x: &CV, y: &mut CV, α: T, β: T);
    fn add_scaled_e(&self, x: &mut CV, α: T);
    fn step_length(&self, dz: &CV, ds: &CV, z: &CV, s: &CV) -> (T, T);
}
