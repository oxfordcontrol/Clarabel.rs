use crate::algebra::*;

//ldl linear solvers kept in a submodule (not flattened)
pub mod ldlsolvers;

//flatten direct KKT module structure
mod datamaps;
mod directldlkktsolver;
mod kkt_assembly;
use datamaps::*;
pub use directldlkktsolver::*;
use kkt_assembly::*;

pub trait DirectLDLSolver<T: FloatT> {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]);
    fn solve(&mut self, x: &mut [T], b: &[T]);
    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool;
    fn required_matrix_shape() -> MatrixTriangle
    where
        Self: Sized;
}
