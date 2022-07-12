use super::*;

use crate::core::{
    cones::{CompositeCone,SupportedCones},
    components::ProblemData};

use clarabel_algebra::*;
use clarabel_timers::*;

pub type DefaultSolver<T = f64> = 
Solver<
    DefaultProblemData<T>,
    DefaultVariables<T>,
    DefaultResiduals<T>,
    DefaultKKTSystem<T>,
    CompositeCone<T>,
    DefaultSolveInfo<T>,
    DefaultSolveResult<T>,
    DefaultSettings<T>,
>;

impl<T: FloatT> DefaultSolver<T> {

pub fn new(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cone_types: &[SupportedCones<T>],
    settings: Settings<T>,
) -> Self {
    let mut timers = Timers::default();
    let mut output;

    timeit! {timers => "setup"; {
    //PJG not clear to me if I should clone or borrow
    //on the types, dims and settings

    let info = DefaultSolveInfo::<T>::new();
    let cones = CompositeCone::<T>::new(cone_types);
    let mut data = DefaultProblemData::<T>::new(P,q,A,b);
    let variables = DefaultVariables::<T>::new(data.n,data.m);
    let residuals = DefaultResiduals::<T>::new(data.n,data.m);

    // equilibrate problem data immediately on setup.
    // this prevents multiple equlibrations if solve!
    // is called more than once.
    timeit!{timers => "equilibrate"; {
        data.equilibrate(&cones,&settings);
    }}

    let kktsystem;
    timeit!{timers => "kktinit"; {
        kktsystem = DefaultKKTSystem::<T>::new(&data,&cones,&settings);
    }}

    // work variables for assembling step direction LHS/RHS
    let step_rhs  = DefaultVariables::<T>::new(data.n,data.m);
    let step_lhs  = DefaultVariables::<T>::new(data.n,data.m);

    // user facing results go here
    //PJG:final argument in julia is the timer.  Made
    //into a separate field here, since it is part of the
    //main solver loop and we shouldn't rely on the user
    //to provide one in a non-default implementation.
    //Change this in Julia.
    let result = DefaultSolveResult::<T>::new(data.m,data.n);

    output = Self{data,variables,residuals,kktsystem,step_lhs,
         step_rhs,info,result,cones,settings,timers: None};

    }} //end "setup" timer.

    //now that the timer is finished we can swap our
    //timer object into the solver structure
    output.timers.replace(timers);

    output
}
}
