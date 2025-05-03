use crate::{algebra::*, solver::core::kktsolvers::HasLinearSolverInfo};

//ldl linear solvers kept in a submodule (not flattened)
pub mod ldlsolvers;

//flatten direct KKT module structure
mod datamaps;
mod directldlkktsolver;
mod kkt_assembly;
use datamaps::*;
pub use directldlkktsolver::*;
use kkt_assembly::*;

pub trait DirectLDLSolverReqs {
    fn required_matrix_shape() -> MatrixTriangle
    where
        Self: Sized;
}
pub trait DirectLDLSolver<T: FloatT>: DirectLDLSolverReqs + HasLinearSolverInfo {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    #[allow(dead_code)] //PJG: could be removed.
    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]);
    fn solve(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]);
    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool;
}
