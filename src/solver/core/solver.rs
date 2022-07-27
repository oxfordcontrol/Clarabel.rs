use super::cones::Cone;
use super::traits::*;
use crate::algebra::*;
use crate::timers::*;

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
// top level solver container type
// ---------------------------------

pub struct Solver<D, V, R, K, C, I, SO, SE> {
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub cones: C,
    pub step_lhs: V,
    pub step_rhs: V,
    pub info: I,
    pub solution: SO,
    pub settings: SE,
    pub timers: Option<Timers>,
}

fn _print_banner(is_verbose: bool) {
    if !is_verbose {
        return;
    }
    const VERSION: &str = env!("CARGO_PKG_VERSION");

    println!("-------------------------------------------------------------");
    println!(
        "           Clarabel.rs v{}  -  Clever Acronym              \n",
        VERSION
    );
    println!("                   (c) Paul Goulart                          ");
    println!("                University of Oxford, 2022                   ");
    println!("-------------------------------------------------------------");
}

// ---------------------------------
// IPSolver trait and its standard implementation.
// ---------------------------------

pub trait IPSolver<T, D, V, R, K, C, I, SO, SE> {
    fn solve(&mut self);
    fn default_start(&mut self);
    fn centering_parameter(&self, α: T) -> T;
}

impl<T, D, V, R, K, C, I, SO, SE> IPSolver<T, D, V, R, K, C, I, SO, SE>
    for Solver<D, V, R, K, C, I, SO, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C>,
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

        //timers is stored as an option so that
        //we can swap it out here and avoid
        //borrow conflicts with other fields.
        let mut timers = self.timers.take().unwrap();

        self.info.reset(&mut timers);

        // solver release info, solver config
        // problem dimensions, cone types etc
        notimeit! {timers; {
            _print_banner(self.settings.core().verbose);
            self.info.print_configuration(&self.settings, &self.data, &self.cones);
            self.info.print_status_header(&self.settings);
        }}

        timeit! {timers => "solve"; {

        // initialize variables to some reasonable starting point
        timeit!{timers => "default start"; {
            self.default_start();
        }}

        timeit!{timers => "IP iteration"; {

        // ----------
        // main loop
        // ----------

        loop {
            //update the residuals
            //--------------
            self.residuals.update(&self.variables, &self.data);

            //calculate duality gap (scaled)
            //--------------
            let μ = self.variables.calc_mu(&self.residuals, &self.cones);

            // convergence check and printing
            // --------------
            self.info.update(
                &self.data,
                &self.variables,
                &self.residuals,&timers);

            let isdone = self.info.check_termination(&self.residuals, &self.settings);

            iter += 1;
            notimeit!{timers; {
                self.info.print_status(&self.settings);
            }}
            if isdone {
                break;
            }

            //
            // update the scalings
            // --------------
            self.variables.scale_cones(&mut self.cones);

            timeit!{timers => "kkt update"; {
                //update the KKT system and the constant
                //parts of its solution
                // --------------
                self.kktsystem.update(&self.data, &self.cones, &self.settings);
            }} // end "kkt update" timer

            // calculate the affine step
            // --------------
            self.step_rhs
                .calc_affine_step_rhs(&self.residuals, &self.variables, &self.cones);

            timeit!{timers => "kkt solve"; {
                self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &self.cones,
                    "affine",
                    &self.settings,
                );
            }}  //end "kkt solve affine" timer

            //calculate step length and centering parameter
            // --------------
            let mut α = self.variables.calc_step_length(&self.step_lhs, &self.cones);
            let σ = self.centering_parameter(α);

            // calculate the combined step and length
            // --------------
            self.step_rhs.calc_combined_step_rhs(
                &self.residuals,
                &self.variables,
                &self.cones,
                &mut self.step_lhs,
                σ,
                μ,
            );

            timeit!{timers => "kkt solve" ; {
                self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &self.cones,
                    "combined",
                    &self.settings,
                );
            }} //end "kkt solve"

            // compute final step length and update the current iterate
            // --------------
            α = self.variables.calc_step_length(&self.step_lhs, &self.cones);
            α *= self.settings.core().max_step_fraction;

            self.variables.add_step(&self.step_lhs, α);

            //record scalar values from this iteration
            self.info.save_scalars(μ, α, σ, iter);

        } //end loop
        // ----------
        // ----------

        }} //end "IP iteration" timer

        }} // end "solve" timer

        //store final solution, timing etc
        self.info.finalize(&mut timers);
        self.solution
            .finalize(&self.data, &self.variables, &self.info);

        //stow the timers back into Option in the solver struct
        self.timers.replace(timers);

        self.info.print_footer(&self.settings);
    }

    fn default_start(&mut self) {
        // set all scalings to identity (or zero for the zero cone)
        self.cones.set_identity_scaling();
        // Refactor
        self.kktsystem
            .update(&self.data, &self.cones, &self.settings);
        // solve for primal/dual initial points via KKT
        self.kktsystem
            .solve_initial_point(&mut self.variables, &self.data, &self.settings);
        // fix up (z,s) so that they are in the cone
        self.variables.shift_to_cone(&self.cones);
    }

    fn centering_parameter(&self, α: T) -> T {
        T::powi(T::one() - α, 3)
    }
}
