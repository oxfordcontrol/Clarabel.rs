#![allow(non_snake_case)]

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
    DefaultResiduals<T>,
    DefaultKKTSystem<T>,
    DefaultSolveInfo<T>,
    DefaultSolveResult<T>,
    ConeSet<T>,
    Settings<T>,
>;


impl<T:FloatT> DefaultSolver<T>
{

    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cone_types: &[SupportedCones],
        cone_dims: &[usize],
        settings: Settings<T>
    ) -> Self
    {
        //PJG not clear to me if I should clone or borrow
        //on the types, dims and settings

        let info = DefaultSolveInfo::<T>::new();
        let cones = ConeSet::<T>::new(&cone_types,&cone_dims);
        let mut data = DefaultProblemData::<T>::new(P,q,A,b,&cones);
        let variables = DefaultVariables::<T>::new(data.n,&cones);
        let residuals = DefaultResiduals::<T>::new(data.n,data.m);

        // equilibrate problem data immediately on setup.
        // this prevents multiple equlibrations if solve!
        // is called more than once.
        data.equilibrate(&cones,&settings);

        let kktsystem = DefaultKKTSystem::<T>::new(&cones,&settings);

        // work variables for assembling step direction LHS/RHS
        let step_rhs  = DefaultVariables::<T>::new(data.n,&cones);
        let step_lhs  = DefaultVariables::<T>::new(data.n,&cones);

        // user facing results go here
        //PJG:final argument in julia is the timer.  Left out
        //here for now.
        let result  = DefaultSolveResult::<T>::new(data.m,data.n);

        Self{data: data,variables,residuals,kktsystem,step_lhs,
             step_rhs,info,result,cones,settings: settings}

    }
}
