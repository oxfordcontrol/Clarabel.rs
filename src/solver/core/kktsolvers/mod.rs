use super::{cones::CompositeCone, CoreSettings};
use crate::algebra::*;

pub mod direct;

pub trait KKTSolver<T: FloatT> {
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool;
    fn setrhs(&mut self, x: &[T], z: &[T]);
    fn solve(
        &mut self,
        x: Option<&mut [T]>,
        z: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool;
}
