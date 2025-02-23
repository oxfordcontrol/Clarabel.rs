#![allow(non_snake_case)]
use super::{cones::CompositeCone, CoreSettings};
use crate::algebra::*;

pub mod direct;

pub trait KKTSolver<T: FloatT>: HasLinearSolverInfo {
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool;
    fn setrhs(&mut self, x: &[T], z: &[T]);
    fn solve(
        &mut self,
        x: Option<&mut [T]>,
        z: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool;
    fn update_P(&mut self, P: &CscMatrix<T>);
    fn update_A(&mut self, A: &CscMatrix<T>);
}

pub trait HasLinearSolverInfo {
    fn linear_solver_info(&self) -> LinearSolverInfo;
}
#[derive(Debug, Default, Clone)]
pub struct LinearSolverInfo {
    pub name: String,
    pub threads: usize,
    pub direct: bool,
    pub nnzA: usize, // nnz in A = LDL^T
    pub nnzL: usize, // nnz in P = LDL^T
}
