use crate::algebra::*;
use crate::*;

pub struct Solver<D, V, R, K, SI, SR, C, S> {
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub step_lhs: V,
    pub step_rhs: V,
    pub info: SI,
    pub result: SR,
    pub cones: C,
    pub settings: S,
}

pub trait IPSolver<T, D, V, R, K, SI, SR, C> {
    fn solve(&mut self);
    fn default_start(&mut self);
    fn centering_parameter(&self, α: T) -> T;
}

// PJG: Make Settings a trait and implement a
// default settings type.   The trait should only
// serve up basic values via getters and have no
// other methods, so should be reusable across
// problem format implementations


impl<T, D, V, R, K, SI, SR, C> IPSolver<T, D, V, R, K, SI, SR,C>
    for Solver<D, V, R, K, SI, SR, C, Settings<T>>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C>,
    SI: SolveInfo<T, D = D, V = V, R = R, C = C>,
    SR: SolveResult<T, D = D, V = V, SI = SI>,
    C : Cone<T>,
{
    fn solve(&mut self) {
        let s = self;

        // various initializations
        s.info.reset();
        let mut iter: u32 = 0;
        // timer  = s.info.timer //PJG fix
        //

        // solver release info, solver config
        // problem dimensions, cone types etc
        // @notimeit //PJG fix
        s.info.print_header(&s.settings, &s.data, &s.cones);

        // @timeit timer "solve!" begin //PJG fix

        // initialize variables to some reasonable starting point
        //  @timeit_debug timer "default start" //PJG fix
        s.default_start();
        //
        //     @timeit_debug timer "IP iteration" begin
        //
        // ----------
        // main loop
        //----------

        loop {
            //update the residuals
            //--------------
            s.residuals.update(&s.variables, &s.data);

            //calculate duality gap (scaled)
            //--------------
            let μ = s.variables.calc_mu(&s.residuals, &s.cones);

            // convergence check and printing
            // --------------
            //         @timeit_debug timer "check termination" begin
            s.info
                .update(&s.data, &s.variables, &s.residuals);

            let isdone = s.info.check_termination(&s.residuals, &s.settings);
            //         end //end timer

            iter += 1;
            s.info.print_status(&s.settings);
            if isdone {
                break;
            }

            //
            // update the scalings
            // --------------
            // @timeit_debug timer "NT scaling"

            s.variables.scale_cones(&mut s.cones);

            //update the KKT system and the constant
            //parts of its solution
            //--------------
            //         @timeit_debug timer "kkt update"
            s.kktsystem.update(&s.data, &s.cones);

            // calculate the affine step
            // --------------
            s.step_rhs
                .calc_affine_step_rhs(&s.residuals, &s.variables, &s.cones);
            //
            //         @timeit_debug timer "kkt solve" begin
            s.kktsystem.solve(
                &mut s.step_lhs,
                &s.step_rhs,
                &s.data,
                &s.variables,
                &s.cones,
                "affine",
            );
            //end

            //calculate step length and centering parameter
            // --------------
            let mut α = s.variables.calc_step_length(&s.step_lhs, &s.cones);
            let σ = s.centering_parameter(α);

            // calculate the combined step and length
            // --------------
            s.step_rhs.calc_combined_step_rhs(
                &s.residuals,
                &s.variables,
                &s.cones,
                &mut s.step_lhs,
                σ,
                μ,
            );

            //@timeit_debug timer "kkt solve" begin
            s.kktsystem.solve(
                &mut s.step_lhs,
                &s.step_rhs,
                &s.data,
                &s.variables,
                &s.cones,
                "combined",
            );
            //end //end timer

            // compute final step length and update the current iterate
            // --------------
            //         @timeit_debug timer "step length"
            α = s.variables.calc_step_length(&s.step_lhs, &s.cones);
            α *= s.settings.max_step_fraction;

            s.variables.add_step(&s.step_lhs, α);

            //record scalar values from this iteration
            s.info.save_scalars(μ, α, σ, iter);
        }
        // ----------
        // ----------

        // end IP iteration timer

        // end solve! timer

        s.info.finalize(); //halts timers

        s.result.finalize(&s.data, &s.variables, &s.info);
        //
        // @notimeit
        s.info.print_footer(&s.settings);
        //
        //PJG: not clear if I am returning a result or
        //what here.
        // return s.result
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
