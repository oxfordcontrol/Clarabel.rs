pub use crate::cones::*;
pub use crate::algebra::*;
pub mod direct_quasidefinite;

pub trait KKTSolver<'a, T: FloatT> {
    fn update(&mut self, cones: ConeSet<T>);
    fn setrhs(&mut self, x: &[T], z: &[T]);
    fn solve(&mut self,  x: Option<&mut [T]>, z: Option<&mut [T]>);
}
