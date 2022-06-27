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

use core::hash::{Hash, Hasher};
use std::{cmp::PartialEq, mem::discriminant};

#[derive(Debug, Clone, Copy)]
pub enum SupportedCones<T> {
    ZeroConeT(usize),        // params: cone_dim
    NonnegativeConeT(usize), // params: cone_dim
    SecondOrderConeT(usize), // params: cone_dim
    PlaceHolderT(usize, T),  // params: cone_dim, exponent
}

//PJG: is there a more compact way of displaying only the variant name?
impl<T: FloatT> std::fmt::Display for SupportedCones<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let str = match self {
            SupportedCones::ZeroConeT(_) => "ZeroConeT",
            SupportedCones::NonnegativeConeT(_) => "NonnegativeConeT",
            SupportedCones::SecondOrderConeT(_) => "SecondOrderConeT",
            SupportedCones::PlaceHolderT(_, _) => "PlaceHolderConeT",
        };
        write!(f, "{}", str)
    }
}

// we will use the SupportedCones as a user facing marker
// for the constraint types, and then map them through
// a dictionary to get the internal cone representations.
// we will also make a HashMap of cone type counts, so need
// to define custom hashing and comparator ops
impl<T> Eq for SupportedCones<T> {}
impl<T> PartialEq for SupportedCones<T> {
    fn eq(&self, other: &Self) -> bool {
        discriminant(self) == discriminant(other)
    }
}

impl<T> Hash for SupportedCones<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);
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
