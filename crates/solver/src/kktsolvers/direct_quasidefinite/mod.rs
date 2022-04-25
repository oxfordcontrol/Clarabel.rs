pub mod datamap;
pub mod directquasidefinitekktsolver;
pub mod ldlsolvers;
pub mod utils;
pub use utils::*;
pub use crate::algebra::*;

pub trait DirectLDLSolver<T:FloatT> {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn offset_values(&mut self, index: &[usize], values: T);
    fn solve(&mut self, x: &mut [T], b: &[T]);
    fn refactor(&mut self);
}
