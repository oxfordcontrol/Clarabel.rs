pub mod datamap;
pub mod directquasidefinitekktsolver;
pub mod ldlsolvers;
pub mod utils;


//PJG: Should I really re-export here?
pub use datamap::*;
pub use directquasidefinitekktsolver::*;
pub use ldlsolvers::*;
pub use utils::*;

use clarabel_algebra::*;
use std::ops::Range; 

//PJG: includes WTF

pub trait DirectLDLSolver<T: FloatT> {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    fn offset_diagonal(&mut self, index: Range<usize>, values: T, signs: &[i8]);
    fn solve(&mut self, x: &mut [T], b: &[T]);
    fn refactor(&mut self, kkt: &CscMatrix<T>);
}
