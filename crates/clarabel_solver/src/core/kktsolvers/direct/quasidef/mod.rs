use clarabel_algebra::*;

//ldl linear solvers kept in a submodule (not flattened)
pub mod ldlsolvers;

//flatten direct KKT module structure
mod datamap;
mod directldlkktsolver;
mod utils;
pub use datamap::*;
pub use directldlkktsolver::*;
pub use utils::*;

pub trait DirectLDLSolver<T: FloatT> {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]);
    fn solve(&mut self, x: &mut [T], b: &[T]);
    fn refactor(&mut self, kkt: &CscMatrix<T>);
    fn required_matrix_shape() -> MatrixTriangle
    where
        Self: Sized;
}
