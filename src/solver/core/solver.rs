use self::internal::*;
use super::cones::Cone;
use super::traits::*;
use crate::algebra::*;
use crate::timers::*;

// ---------------------------------
// Solver status type
// ---------------------------------

/// Status of solver at termination

#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum SolverStatus {
    /// Problem is not solved (solver hasn't run).
    Unsolved,
    /// Solver terminated with a solution.
    Solved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility.
    PrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility.
    DualInfeasible,
    /// Solver terminated with a solution (reduced accuracy)
    AlmostSolved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility (reduced accuracy).
    AlmostPrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility (reduced accuracy).
    AlmostDualInfeasible,
    /// Iteration limit reached before solution or infeasibility certificate found.
    MaxIterations,
    /// Time limit reached before solution or infeasibility certificate found.
    MaxTime,
    /// Solver terminated with a numerical error
    NumericalError,
    /// Solver terminated due to lack of progress.
    InsufficientProgress,
}

impl SolverStatus {
    pub(crate) fn is_infeasible(&self) -> bool {
        matches!(
            *self,
            |SolverStatus::PrimalInfeasible| SolverStatus::DualInfeasible
                | SolverStatus::AlmostPrimalInfeasible
                | SolverStatus::AlmostDualInfeasible
        )
    }

    pub(crate) fn is_errored(&self) -> bool {
        // status is any of the error codes
        matches!(
            *self,
            SolverStatus::NumericalError | SolverStatus::InsufficientProgress
        )
    }
}

#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum StepDirection {
    Affine,
    Combined,
}

/// Scaling strategy used by the solver when
/// linearizing centrality conditions.  
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum ScalingStrategy {
    PrimalDual,
    Dual,
}

/// An enum for reporting strategy checkpointing
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
enum StrategyCheckpoint {
    Update(ScalingStrategy), // Checkpoint is suggesting a new ScalingStrategy
    NoUpdate,                // Checkpoint recommends no change to ScalingStrategy
    Fail,                    // Checkpoint found a problem but no more ScalingStrategies to try
}

impl std::fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Default for SolverStatus {
    fn default() -> Self {
        SolverStatus::Unsolved
    }
}

// ---------------------------------
// top level solver container type
// ---------------------------------

// The top-level solver.

// This trait is defined with a collection of mutually interacting associated types.
// See the [`DefaultSolver`](crate::solver::implementations::default) for an example.

pub struct Solver<D, V, R, K, C, I, SO, SE> {
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub cones: C,
    pub step_lhs: V,
    pub step_rhs: V,
    pub prev_vars: V,
    pub info: I,
    pub solution: SO,
    pub settings: SE,
    pub timers: Option<Timers>,
}

fn _print_banner(is_verbose: bool) {
    if !is_verbose {
        return;
    }

    println!("-------------------------------------------------------------");
    println!(
        "           Clarabel.rs v{}  -  Clever Acronym              \n",
        crate::VERSION
    );
    println!("                   (c) Paul Goulart                          ");
    println!("                University of Oxford, 2022                   ");
    println!("-------------------------------------------------------------");
}

// ---------------------------------
// IPSolver trait and its standard implementation.
// ---------------------------------

/// An interior point solver implementing a predictor-corrector scheme

// Only the main solver function lives in IPSolver, since this is the
// only publicly facing trait we want to give the solver.   Additional
// internal functionality for the top level solver object is implemented
// for the IPSolverUtilities trait below, upon which IPSolver depends

pub trait IPSolver<T, D, V, R, K, C, I, SO, SE> {
    /// Run the solver
    fn solve(&mut self);
}

impl<T, D, V, R, K, C, I, SO, SE> IPSolver<T, D, V, R, K, C, I, SO, SE>
    for Solver<D, V, R, K, C, I, SO, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C, SE = SE>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
    C: Cone<T>,
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SO: Solution<T, D = D, V = V, I = I>,
    SE: Settings<T>,
{
    fn solve(&mut self) {
        // various initializations
        let mut iter: u32 = 0;
        let mut σ = T::one();
        let mut α = T::zero();
        let mut μ;

        //timers is stored as an option so that
        //we can swap it out here and avoid
        //borrow conflicts with other fields.
        let mut timers = self.timers.take().unwrap();

        // solver release info, solver config
        // problem dimensions, cone types etc
        notimeit! {timers; {
            _print_banner(self.settings.core().verbose);
            self.info.print_configuration(&self.settings, &self.data, &self.cones);
            self.info.print_status_header(&self.settings);
        }}

        self.info.reset(&mut timers);

        timeit! {timers => "solve"; {

        // initialize variables to some reasonable starting point
        timeit!{timers => "default start"; {
            self.default_start();
        }}

        timeit!{timers => "IP iteration"; {

        // ----------
        // main loop
        // ----------

        let mut scaling = {
            if self.cones.allows_primal_dual_scaling() {ScalingStrategy::PrimalDual}
            else {ScalingStrategy::Dual}
        };

        loop {

            //update the residuals
            //--------------
            self.residuals.update(&self.variables, &self.data);

            //calculate duality gap (scaled)
            //--------------
            μ = self.variables.calc_mu(&self.residuals, &self.cones);

            // record scalar values from most recent iteration.
            // This captures μ at iteration zero.
            self.info.save_scalars(μ, α, σ, iter);

            // convergence check and printing
            // --------------
            self.info.update(
                &self.data,
                &self.variables,
                &self.residuals,&timers);

            notimeit!{timers; {
                self.info.print_status(&self.settings);
            }}

            let isdone = self.info.check_termination(&self.residuals, &self.settings, iter);

            // check for termination due to slow progress and update strategy
            if isdone{
                    match self.strategy_checkpoint_insufficient_progress(scaling){
                        StrategyCheckpoint::NoUpdate | StrategyCheckpoint::Fail => {break}
                        StrategyCheckpoint::Update(s) => {scaling = s; continue}
                    }
            }  // allows continuation if new strategy provided


            // update the scalings
            // --------------
            let is_scaling_success = self.variables.scale_cones(&mut self.cones,μ,scaling);
            // check whether variables are interior points
            match self.strategy_checkpoint_is_scaling_success(is_scaling_success,scaling){
                StrategyCheckpoint::Fail => {break}
                StrategyCheckpoint::NoUpdate => {} // we only expect NoUpdate or Fail here
                StrategyCheckpoint::Update(_) => {unreachable!()}
            }

            //increment counter here because we only count
            //iterations that produce a KKT update
            iter += 1;

            // Update the KKT system and the constant parts of its solution.
            // Keep track of the success of each step that calls KKT
            // --------------
            //PJG: This should be a Result in Rust, but needs changes down
            //into the KKT solvers to do that.
            let mut is_kkt_solve_success : bool;
            timeit!{timers => "kkt update"; {
                is_kkt_solve_success = self.kktsystem.update(&self.data, &self.cones, &self.settings);
            }} // end "kkt update" timer

            // calculate the affine step
            // --------------
            self.step_rhs
                .affine_step_rhs(&self.residuals, &self.variables, &self.cones);

            timeit!{timers => "kkt solve"; {
                is_kkt_solve_success = is_kkt_solve_success &&
                self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &mut self.cones,
                    StepDirection::Affine,
                    &self.settings,
                );
            }}  //end "kkt solve affine" timer

            // combined step only on affine step success
            if is_kkt_solve_success {

                //calculate step length and centering parameter
                // --------------
                α = self.get_step_length(StepDirection::Affine, scaling);
                σ = self.centering_parameter(α);

                // make a reduced Mehrotra correction in the first iteration
                // to accommodate badly centred starting points
                let m = if iter > 1 {T::one()} else {α};

                // calculate the combined step and length
                // --------------
                self.step_rhs.combined_step_rhs(
                    &self.residuals,
                    &self.variables,
                    &mut self.cones,
                    &mut self.step_lhs,
                    σ,
                    μ,
                    m
                );

                timeit!{timers => "kkt solve" ; {
                    is_kkt_solve_success =
                    self.kktsystem.solve(
                        &mut self.step_lhs,
                        &self.step_rhs,
                        &self.data,
                        &self.variables,
                        &mut self.cones,
                        StepDirection::Combined,
                        &self.settings,
                    );
                }} //end "kkt solve"
            }

            // check for numerical failure and update strategy
            match self.strategy_checkpoint_numerical_error(is_kkt_solve_success,scaling) {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }


            // compute final step length and update the current iterate
            // --------------
            α = self.get_step_length(StepDirection::Combined,scaling);

            // check for undersized step and update strategy
            match self.strategy_checkpoint_small_step(α, scaling) {
                StrategyCheckpoint::NoUpdate => {}
                StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                StrategyCheckpoint::Fail => {α = T::zero(); break}
            }

            // Copy previous iterate in case the next one is a dud
            self.info.save_prev_iterate(&self.variables,&mut self.prev_vars);

            self.variables.add_step(&self.step_lhs, α);

        } //end loop
        // ----------
        // ----------

        }} //end "IP iteration" timer

        }} // end "solve" timer

        // Check we if actually took a final step.  If not, we need
        // to recapture the scalars and print one last line
        if α == T::zero() {
            self.info.save_scalars(μ, α, σ, iter);
            notimeit! {timers; {self.info.print_status(&self.settings);}}
        }

        //store final solution, timing etc
        self.info
            .finalize(&self.residuals, &self.settings, &mut timers);

        self.solution
            .finalize(&self.data, &self.variables, &self.info);

        self.info.print_footer(&self.settings);

        //stow the timers back into Option in the solver struct
        self.timers.replace(timers);
    }
}

// Encapsulate the internal helpers trait in a private module
// so it doesn't get exported
mod internal {
    use super::super::cones::Cone;
    use super::super::traits::*;
    use super::*;

    pub(super) trait IPSolverInternals<T, D, V, R, K, C, I, SO, SE> {
        /// Find an initial condition
        fn default_start(&mut self);

        /// Compute a centering parameter
        fn centering_parameter(&self, α: T) -> T;

        /// Compute the current step length
        fn get_step_length(&mut self, step_direction: StepDirection, scaling: ScalingStrategy)
            -> T;

        /// backtrack a step direction to the barrier
        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T;

        /// Scaling strategy checkpointing functions
        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;
    }

    impl<T, D, V, R, K, C, I, SO, SE> IPSolverInternals<T, D, V, R, K, C, I, SO, SE>
        for Solver<D, V, R, K, C, I, SO, SE>
    where
        T: FloatT,
        D: ProblemData<T, V = V>,
        V: Variables<T, D = D, R = R, C = C, SE = SE>,
        R: Residuals<T, D = D, V = V>,
        K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
        C: Cone<T>,
        I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
        SO: Solution<T, D = D, V = V, I = I>,
        SE: Settings<T>,
    {
        fn default_start(&mut self) {
            if self.cones.is_symmetric() {
                // set all scalings to identity (or zero for the zero cone)
                self.cones.set_identity_scaling();
                // Refactor
                self.kktsystem
                    .update(&self.data, &self.cones, &self.settings);
                // solve for primal/dual initial points via KKT
                self.kktsystem
                    .solve_initial_point(&mut self.variables, &self.data, &self.settings);
                // fix up (z,s) so that they are in the cone
                self.variables.symmetric_initialization(&mut self.cones);
            } else {
                // Assigns unit (z,s) and zeros the primal variables
                self.variables.unit_initialization(&self.cones);
            }
        }

        fn centering_parameter(&self, α: T) -> T {
            T::powi(T::one() - α, 3)
        }

        fn get_step_length(
            &mut self,
            step_direction: StepDirection,
            scaling: ScalingStrategy,
        ) -> T {
            //step length to stay within the cones
            let mut α = self.variables.calc_step_length(
                &self.step_lhs,
                &mut self.cones,
                &self.settings,
                step_direction,
            );

            // additional barrier function limits for asymmetric cones
            if !self.cones.is_symmetric()
                && step_direction == StepDirection::Combined
                && scaling == ScalingStrategy::Dual
            {
                let αinit = α;
                α = self.backtrack_step_to_barrier(αinit);
            }
            α
        }

        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T {
            let step = self.settings.core().linesearch_backtrack_step;
            let mut α = αinit;

            for _ in 0..50 {
                let barrier = self.variables.barrier(&self.step_lhs, α, &mut self.cones);
                if barrier < T::one() {
                    return α;
                } else {
                    α = step * α;
                }
            }
            α
        }

        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            if self.info.get_status() != SolverStatus::InsufficientProgress {
                // there is no problem, so nothing to do
                output = StrategyCheckpoint::NoUpdate;
            } else {
                // recover old iterate since "insufficient progress" often
                // involves actual degradation of results
                self.info
                    .reset_to_prev_iterate(&mut self.variables, &self.prev_vars);

                // If problem is asymmetric, we can try to continue with the dual-only strategy
                if !self.cones.is_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
                    self.info.set_status(SolverStatus::Unsolved);
                    output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
                } else {
                    output = StrategyCheckpoint::Fail;
                }
            }
            output
        }

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            // No update if kkt updates successfully
            if is_kkt_solve_success {
                output = StrategyCheckpoint::NoUpdate;
            }
            // If problem is asymmetric, we can try to continue with the dual-only strategy
            else if !self.cones.is_symmetric() && (scaling == ScalingStrategy::PrimalDual) {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else {
                // out of tricks.  Bail out with an error
                self.info.set_status(SolverStatus::NumericalError);
                output = StrategyCheckpoint::Fail;
            }
            output
        }

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;

            if !self.cones.is_symmetric()
                && scaling == ScalingStrategy::PrimalDual
                && α < self.settings.core().min_switch_step_length
            {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else if α <= T::max(T::zero(), self.settings.core().min_terminate_step_length) {
                self.info.set_status(SolverStatus::InsufficientProgress);
                output = StrategyCheckpoint::Fail;
            } else {
                output = StrategyCheckpoint::NoUpdate;
            }

            output
        }

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            _scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            if is_scaling_success {
                StrategyCheckpoint::NoUpdate
            } else {
                self.info.set_status(SolverStatus::NumericalError);
                StrategyCheckpoint::Fail
            }
        }
    } // end trait impl
} //end internals module
