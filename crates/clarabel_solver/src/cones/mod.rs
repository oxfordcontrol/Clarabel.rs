#![allow(non_snake_case)]
pub mod coneset;
pub mod nonnegativecone;
pub mod socone;
pub mod zerocone;
pub use coneset::*;
pub use nonnegativecone::*;
pub use socone::*;
pub use zerocone::*;

use clarabel_algebra::*;


#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SupportedCones {
    ZeroConeT,
    NonnegativeConeT,
    SecondOrderConeT,
}

impl std::fmt::Display for SupportedCones {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait Cone<T> {
    fn dim(&self) -> usize;
    fn degree(&self) -> usize;
    fn numel(&self) -> usize;
    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool;
    fn WtW_is_diagonal(&self) -> bool;
    fn update_scaling(&mut self, s: &[T], z: &[T]);
    fn set_identity_scaling(&mut self);
    fn λ_circ_λ(&self, x: &mut [T]);
    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]);
    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]);
    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]);
    fn shift_to_cone(&self, z: &mut [T]);
    fn get_WtW_block(&self, WtWblock: &mut [T]);
    fn gemv_W(&self, is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T);
    fn gemv_Winv(&self, is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T);
    fn add_scaled_e(&self, x: &mut [T], α: T);
    fn step_length(&self, dz: &[T], ds: &[T], z: &[T], s: &[T]) -> (T, T);
}


