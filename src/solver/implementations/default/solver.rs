use super::*;
use crate::solver::core::{
    cones::{CompositeCone, SupportedConeT},
    traits::ProblemData,
    Solver,
};

use crate::algebra::*;
use crate::timers::*;

/// Solver for problems in standard conic program form

pub type DefaultSolver<T = f64> = Solver<
    DefaultProblemData<T>,
    DefaultVariables<T>,
    DefaultResiduals<T>,
    DefaultKKTSystem<T>,
    CompositeCone<T>,
    DefaultInfo<T>,
    DefaultSolution<T>,
    DefaultSettings<T>,
>;

impl<T> DefaultSolver<T>
where
    T: FloatT,
{
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cone_specs: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Self {
        //sanity check problem dimensions
        _check_dimensions(P, q, A, b, cone_specs);

        let mut timers = Timers::default();
        let mut output;
        let info = DefaultInfo::<T>::new();

        timeit! {timers => "setup"; {

        // reduce the cone sizes.  (A,b) will be reduced
        // within the problem data constructor.  Also makes
        // an internal copy of the user cone specification
        let presolver = Presolver::<T>::new(A,b,cone_specs,&settings);

        let cones = CompositeCone::<T>::new(&presolver.cone_specs);
        let mut data = DefaultProblemData::<T>::new(P,q,A,b,presolver);

        let variables = DefaultVariables::<T>::new(data.n,data.m);
        let residuals = DefaultResiduals::<T>::new(data.n,data.m);

        // equilibrate problem data immediately on setup.
        // this prevents multiple equlibrations if solve!
        // is called more than once.
        timeit!{timers => "equilibration"; {
            data.equilibrate(&cones,&settings);
        }}

        let kktsystem;
        timeit!{timers => "kktinit"; {
            kktsystem = DefaultKKTSystem::<T>::new(&data,&cones,&settings);
        }}

        // work variables for assembling step direction LHS/RHS
        let step_rhs  = DefaultVariables::<T>::new(data.n,data.m);
        let step_lhs  = DefaultVariables::<T>::new(data.n,data.m);
        let prev_vars = DefaultVariables::<T>::new(data.n,data.m);

        // user facing results go here.
        let solution = DefaultSolution::<T>::new(data.presolver.mfull,data.n);

        output = Self{data,variables,residuals,kktsystem,step_lhs,
             step_rhs,prev_vars,info,solution,cones,settings,timers: None};

        }} //end "setup" timer.

        //now that the timer is finished we can swap our
        //timer object into the solver structure
        output.timers.replace(timers);

        output
    }
}

fn _check_dimensions<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cone_types: &[SupportedConeT<T>],
) {
    let m = b.len();
    let n = q.len();
    let p = cone_types.iter().fold(0, |acc, cone| acc + cone.nvars());

    assert!(m == A.nrows(), "A and b incompatible dimensions.");
    assert!(
        p == m,
        "Constraint dimensions inconsistent with size of cones."
    );
    assert!(n == A.ncols(), "A and q incompatible dimensions.");
    assert!(n == P.ncols(), "P and q incompatible dimensions.");
    assert!(P.is_square(), "P not square.");
}
