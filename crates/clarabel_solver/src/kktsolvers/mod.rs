use crate::cones::*;
use clarabel_algebra::*;
pub mod direct;

pub trait KKTSolver<T: FloatT> {
    fn update(&mut self, cones: &CompositeCone<T>);
    fn setrhs(&mut self, x: &[T], z: &[T]);
    fn solve(&mut self, x: Option<&mut [T]>, z: Option<&mut [T]>);
}
