use clarabel_algebra::*;
use std::ops::Range; 

//ldl linear solvers kept in a submodule (not flattened) 
pub mod ldlsolvers;

//flatten direct KKT module structure
mod datamap;
mod directquasidefinitekktsolver;
mod utils;
pub use directquasidefinitekktsolver::*;
pub use datamap::*;
pub use utils::*;

pub trait DirectLDLSolver<T: FloatT> {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    fn offset_diagonal(&mut self, index: Range<usize>, values: T, signs: &[i8]);
    fn solve(&mut self, x: &mut [T], b: &[T]);
    fn refactor(&mut self, kkt: &CscMatrix<T>);
}
