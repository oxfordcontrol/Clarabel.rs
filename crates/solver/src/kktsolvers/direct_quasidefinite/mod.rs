pub mod datamap;
pub mod directquasidefinitekktsolver;
pub mod ldlsolvers;
pub mod utils;
pub use crate::algebra::*;
use crate::Settings;
pub use utils::*;

pub trait DirectLDLSolver<T: FloatT> {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    fn offset_values(&mut self, index: &[usize], values: T);
    fn solve(&mut self, x: &mut [T], b: &[T], settings: &Settings<T>);
    fn refactor(&mut self);
}
