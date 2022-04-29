pub mod equilibration;
pub mod kktsystem;
pub mod problemdata;
pub mod residuals;
pub mod solveinfo;
pub mod solveresult;
pub mod variables;

pub use super::*;
pub use crate::solver::Solver;
pub use crate::ConeSet;
pub use crate::Settings;
pub use equilibration::*;
pub use kktsystem::*;
pub use problemdata::*;
pub use residuals::*;
pub use solveinfo::*;
pub use solveresult::*;
pub use variables::*;

type DefaultSolver<T = f64> = Solver<
    DefaultProblemData<T>,
    DefaultVariables<T>,
    ConeSet<T>,
    DefaultKKTSystem<T>,
    DefaultResiduals<T>,
    DefaultSolveInfo<T>,
    DefaultSolveResult<T>,
    Settings<T>,
>;
