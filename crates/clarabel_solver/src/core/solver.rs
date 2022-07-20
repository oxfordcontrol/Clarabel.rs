use super::cones::Cone;
use super::traits::*;
use clarabel_timers::*;
use clarabel_algebra::*;


// ---------------------------------
// Solver status type 
// ---------------------------------

#[derive(PartialEq, Clone, Debug)]
pub enum SolverStatus {
    Unsolved,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    MaxIterations,
    MaxTime,
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
// Solver type and standard implemention 
// ---------------------------------


pub struct Solver<D, V, R, K, C, SI, SR, SE> {
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub cones: C,
    pub step_lhs: V,
    pub step_rhs: V,
    pub info: SI,
    pub result: SR,
    pub settings: SE,
    pub timers: Option<Timers>,
}

pub trait IPSolver<T, D, V, R, K, C, SI, SR, SE> {
    
    fn solve(&mut self);
    fn default_start(&mut self);
    fn centering_parameter(&self, α: T) -> T;
}


impl<T, D, V, R, K, C, SI, SR, SE> IPSolver<T, D, V, R, K, C, SI, SR, SE>
    for Solver<D, V, R, K, C, SI, SR, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C>,
    C: Cone<T>,
    SI: SolveInfo<T, D = D, V = V, R = R, C = C, SE = SE>,
    SR: SolveResult<T, D = D, V = V, SI = SI>,
    SE: Settings<T>,
{
    fn solve(&mut self) {

        let s = self;

        // various initializations
        s.info.reset();
        let mut iter: u32 = 0;

        //timers is stored as an option so that
        //we can swap it out here and avoid
        //borrow conflicts with other fields.
        let mut timers = s.timers.take().unwrap();
        // reset the "solve" timer, but keep the "setup"
        timers.reset_timer("solve");

        // solver release info, solver config
        // problem dimensions, cone types etc
        // @notimeit //PJG fix
        s.info.print_header(&s.settings, &s.data, &s.cones);

        timeit! {timers => "solve"; {

        // initialize variables to some reasonable starting point
        timeit!{timers => "default start"; {
            s.default_start();
        }}

        timeit!{timers => "IP iteration"; {

        // ----------
        // main loop
        // ----------

        loop {
            //update the residuals
            //--------------
            timeit!{timers => "residuals update"; {
            s.residuals.update(&s.variables, &s.data);
            }}

            //calculate duality gap (scaled)
            //--------------
            let μ;
            timeit!{timers => "calc_mu"; {
            μ = s.variables.calc_mu(&s.residuals, &s.cones);
            }}

            // convergence check and printing
            // --------------
            let isdone;
            timeit!{timers => "check termination"; {
            s.info
                .update(&s.data, &s.variables, &s.residuals);

            isdone = s.info.check_termination(&s.residuals, &s.settings);
            }} //end "check termination" timer

            iter += 1;
            notimeit!{timers; {
            s.info.print_status(&s.settings);
            }}
            if isdone {
                break;
            }

            //
            // update the scalings
            // --------------
            timeit!{timers => "NT scaling"; {
                s.variables.scale_cones(&mut s.cones);
            }}

            timeit!{timers => "kkt update"; {
            //update the KKT system and the constant
            //parts of its solution
            // --------------
            s.kktsystem.update(&s.data, &s.cones);
            }} // end "kkt update" timer

            // calculate the affine step
            // --------------
            timeit!{timers => "calc_affine_step_rhs"; {
            s.step_rhs
                .calc_affine_step_rhs(&s.residuals, &s.variables, &s.cones);
            }}

            timeit!{timers => "kkt solve affine"; {
            s.kktsystem.solve(
                &mut s.step_lhs,
                &s.step_rhs,
                &s.data,
                &s.variables,
                &s.cones,
                "affine",
            );
            }}  //end "kkt solve affine" timer

            //calculate step length and centering parameter
            // --------------
            let mut α;
            let σ;
            timeit!{timers => "step length affine"; {
            α = s.variables.calc_step_length(&s.step_lhs, &s.cones);
            σ = s.centering_parameter(α);
            }}

            // calculate the combined step and length
            // --------------
            timeit!{timers => "calc_combined_step_rhs"; {
            s.step_rhs.calc_combined_step_rhs(
                &s.residuals,
                &s.variables,
                &s.cones,
                &mut s.step_lhs,
                σ,
                μ,
            );
            }}

            timeit!{timers => "kkt solve combined" ; {
            s.kktsystem.solve(
                &mut s.step_lhs,
                &s.step_rhs,
                &s.data,
                &s.variables,
                &s.cones,
                "combined",
            );
            }} //end "kkt solve"

            // compute final step length and update the current iterate
            // --------------
            timeit!{timers => "final step length and add"; {
            α = s.variables.calc_step_length(&s.step_lhs, &s.cones);
            α *= s.settings.core().max_step_fraction;

            s.variables.add_step(&s.step_lhs, α);
            }} //end "IP step" timer

            //record scalar values from this iteration
            timeit!{timers => "save scalars"; {
            s.info.save_scalars(μ, α, σ, iter);
            }}
        } //end loop
        // ----------
        // ----------

        }} //end "IP iteration" timer

        }} // end "solve" timer

        //store final solution, timing etc
        s.result.finalize(&s.data, &s.variables, &s.info);
        s.info.finalize(&timers);

        //stow the timers back into Option in the solver struct
        s.timers.replace(timers);

        s.info.print_footer(&s.settings);

    }

    fn default_start(&mut self) {
        let s = self;

        // set all scalings to identity (or zero for the zero cone)
        s.cones.set_identity_scaling();
        // Refactor
        s.kktsystem.update(&s.data, &s.cones);
        // solve for primal/dual initial points via KKT
        s.kktsystem.solve_initial_point(&mut s.variables, &s.data);
        // fix up (z,s) so that they are in the cone
        s.variables.shift_to_cone(&s.cones);
    }

    fn centering_parameter(&self, α: T) -> T {
        T::powi(T::one() - α, 3)
    }
}
