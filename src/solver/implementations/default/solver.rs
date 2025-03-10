use super::*;
use crate::{
    io::ConfigurablePrintTarget,
    solver::core::{
        cones::{CompositeCone, SupportedConeT},
        kktsolvers::HasLinearSolverInfo,
        traits::ProblemData,
        Solver,
    },
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
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Self {
        //sanity check problem dimensions
        _check_dimensions(P, q, A, b, cones);

        let mut timers = Timers::default();
        let mut output;
        let mut info = DefaultInfo::<T>::new();

        timeit! {timers => "setup"; {

        // user facing results go here.
        let solution = DefaultSolution::<T>::new(A.n, A.m);

        // presolve / chordal decomposition if needed,
        // then take an internal copy of the problem data
        let mut data;
        timeit!{timers => "presolve"; {
            data = DefaultProblemData::<T>::new(P,q,A,b,cones,&settings);
        }}

        let cones = CompositeCone::<T>::new(&data.cones);
        assert_eq!(cones.numel, data.m);
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
        info.linsolver = kktsystem.linear_solver_info();

        // work variables for assembling step direction LHS/RHS
        let step_rhs  = DefaultVariables::<T>::new(data.n,data.m);
        let step_lhs  = DefaultVariables::<T>::new(data.n,data.m);
        let prev_vars = DefaultVariables::<T>::new(data.n,data.m);

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

impl<T> ConfigurablePrintTarget for DefaultSolver<T>
where
    T: FloatT,
{
    fn print_to_stdout(&mut self) {
        self.info.print_to_stdout();
    }
    fn print_to_file(&mut self, file: std::fs::File) {
        self.info.print_to_file(file)
    }
    fn print_to_stream(&mut self, stream: Box<dyn std::io::Write + Send + Sync>) {
        self.info.print_to_stream(stream)
    }
    fn print_to_sink(&mut self) {
        self.info.print_to_sink()
    }
    fn print_to_buffer(&mut self) {
        self.info.print_to_buffer();
    }
    fn get_print_buffer(&mut self) -> std::io::Result<String> {
        self.info.get_print_buffer()
    }
}
