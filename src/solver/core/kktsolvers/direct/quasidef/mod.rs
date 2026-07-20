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

    /// Solve with iterative refinement performed inside the backend,
    /// against the backend's internal (unregularized) matrix copy.
    /// Backends that maintain an internally permuted copy can run the
    /// whole refinement loop in permuted coordinates, avoiding the
    /// per-backsolve permutations of repeated `solve` calls.   Returns
    /// None if the backend does not support this, in which case the
    /// caller performs its own refinement via repeated `solve` calls.
    #[allow(clippy::too_many_arguments)]
    fn solve_refined(
        &mut self,
        _x: &mut [T],
        _b: &[T],
        _reltol: T,
        _abstol: T,
        _max_iter: u32,
        _stop_ratio: T,
    ) -> Option<bool> {
        None
    }
}
