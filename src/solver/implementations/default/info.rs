use super::*;
use crate::algebra::*;
use crate::io::PrintTarget;
use crate::solver::core::ffi::*;
use crate::solver::core::kktsolvers::LinearSolverInfo;
use crate::solver::core::{traits::Info, SolverStatus};
use crate::solver::traits::Variables;
use crate::timers::*;

/// Standard-form solver type implementing the [`Info`](crate::solver::core::traits::Info) and [`InfoPrint`](crate::solver::core::traits::InfoPrint) traits
#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct DefaultInfo<T> {
    /// interior point path parameter μ
    pub mu: T,
    /// interior point path parameter reduction ratio σ
    pub sigma: T,
    /// step length for the current iteration
    pub step_length: T,
    /// number of iterations
    pub iterations: u32,
    /// primal objective value
    pub cost_primal: T,
    /// dual objective value
    pub cost_dual: T,
    /// primal residual
    pub res_primal: T,
    /// dual residual
    pub res_dual: T,
    /// primal infeasibility residual
    pub res_primal_inf: T,
    /// dual infeasibility residual
    pub res_dual_inf: T,
    /// absolute duality gap
    pub gap_abs: T,
    /// relative duality gap
    pub gap_rel: T,
    /// κ/τ ratio
    pub ktratio: T,

    // previous iterate
    /// primal object value from previous iteration
    pub(crate) prev_cost_primal: T,
    /// dual objective value from previous iteration
    pub(crate) prev_cost_dual: T,
    /// primal residual from previous iteration
    pub(crate) prev_res_primal: T,
    /// dual residual from previous iteration
    pub(crate) prev_res_dual: T,
    /// absolute duality gap from previous iteration
    pub(crate) prev_gap_abs: T,
    /// relative duality gap from previous iteration
    pub(crate) prev_gap_rel: T,

    // best iterate seen so far (by worst-case residual/gap merit), used as a
    // fallback when the solver terminates with a numerical error on a
    // degraded final iterate
    pub(crate) best_cost_primal: T,
    pub(crate) best_cost_dual: T,
    pub(crate) best_res_primal: T,
    pub(crate) best_res_dual: T,
    pub(crate) best_gap_abs: T,
    pub(crate) best_gap_rel: T,
    pub(crate) best_ktratio: T,
    pub(crate) best_merit: T,
    /// solve time
    pub solve_time: f64,
    /// solver status
    pub status: SolverStatus,

    /// linear solver information
    pub linsolver: LinearSolverInfo,

    // target stream for printing
    pub(crate) stream: PrintTarget,
}

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    /// creates a new `DefaultInfo` object
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: FloatT> ClarabelFFI<Self> for DefaultInfo<T> {
    type FFI = super::ffi::DefaultInfoFFI<T>;
}

impl<T> Info<T> for DefaultInfo<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type R = DefaultResiduals<T>;

    fn reset(&mut self, timers: &mut Timers) {
        self.status = SolverStatus::Unsolved;
        self.iterations = 0;
        self.solve_time = 0f64;
        self.best_merit = T::infinity();

        timers.reset_timer("solve");
    }

    fn post_process(&mut self, residuals: &DefaultResiduals<T>, settings: &DefaultSettings<T>) {
        // if there was an error or we ran out of time
        // or iterations, check for partial convergence

        if self.status.is_errored()
            || matches!(self.status, SolverStatus::MaxIterations)
            || matches!(self.status, SolverStatus::MaxTime)
        {
            self.check_convergence_almost(residuals, settings);
        }
    }

    fn finalize(&mut self, timers: &mut Timers) {
        //final check of timers
        self.solve_time = timers.total_time().as_secs_f64();
    }

    fn update(
        &mut self,
        data: &mut DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        timers: &Timers,
    ) {
        // optimality termination check should be computed w.r.t
        // the pre-homogenization x and z variables.
        let τinv = T::recip(variables.τ);

        // unscaled linear term norms
        let normb = data.get_normb();
        let normq = data.get_normq();

        // shortcuts for the equilibration matrices
        let d = &data.equilibration.d;
        let e = &data.equilibration.e;
        let dinv = &data.equilibration.dinv;
        let einv = &data.equilibration.einv;
        let cinv = T::recip(data.equilibration.c);

        // primal and dual costs. dot products are invariant w.r.t
        // equilibration, but we still need to back out the overall
        // objective scaling term c

        let xPx_τinvsq_over2 = residuals.dot_xPx * τinv * τinv / (2.).as_T();
        self.cost_primal = (residuals.dot_qx * τinv + xPx_τinvsq_over2) * cinv;
        self.cost_dual = (-residuals.dot_bz * τinv - xPx_τinvsq_over2) * cinv;

        // variables norms, undoing the equilibration.  Do not unscale
        // by τ yet because the infeasibility residuals are ratios of
        // terms that have no affine parts anyway
        let mut normx = variables.x.norm_scaled(d);
        let mut normz = variables.z.norm_scaled(e) * cinv;
        let mut norms = variables.s.norm_scaled(einv);

        // primal and dual infeasibility residuals.
        self.res_primal_inf = (residuals.rx_inf.norm_scaled(dinv) * cinv) / T::max(T::one(), normz);
        self.res_dual_inf = T::max(
            residuals.Px.norm_scaled(dinv) / T::max(T::one(), normx),
            residuals.rz_inf.norm_scaled(einv) / T::max(T::one(), normx + norms),
        );

        // now back out the τ scaling so we can normalize the unscaled primal / dual errors
        normx *= τinv;
        normz *= τinv;
        norms *= τinv;

        // primal and dual relative residuals.
        self.res_primal =
            residuals.rz.norm_scaled(einv) * τinv / T::max(T::one(), normb + normx + norms);
        self.res_dual =
            residuals.rx.norm_scaled(dinv) * τinv * cinv / T::max(T::one(), normq + normx + normz);

        // absolute and relative gaps
        self.gap_abs = T::abs(self.cost_primal - self.cost_dual);
        self.gap_rel = self.gap_abs
            / T::max(
                T::one(),
                T::min(T::abs(self.cost_primal), T::abs(self.cost_dual)),
            );

        // κ/τ ratio (scaled)
        self.ktratio = variables.κ * τinv;

        // solve time so far (includes setup)
        self.solve_time = timers.total_time().as_secs_f64();
    }

    fn check_termination(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
        iter: u32,
    ) -> bool {
        //  optimality or infeasibility
        // ---------------------
        self.check_convergence_full(residuals, settings);

        //  poor progress
        // ----------------------
        if self.status == SolverStatus::Unsolved
            && iter > 1u32
            && (self.res_dual > self.prev_res_dual || self.res_primal > self.prev_res_primal)
        {
            // Poor progress at high tolerance.
            if self.ktratio < T::epsilon() * (100.).as_T()
                && (self.prev_gap_abs < settings.tol_gap_abs
                    || self.prev_gap_rel < settings.tol_gap_rel)
            {
                self.status = SolverStatus::InsufficientProgress;
            }

            // Going backwards. Stop immediately if residuals diverge out of feasibility tolerance.
            #[allow(clippy::collapsible_if)] // nested if for readability
            if self.ktratio < T::one() {
                if (self.res_dual > settings.tol_feas * (100.).as_T()
                    && self.res_dual > self.prev_res_dual * (100.).as_T())
                    || (self.res_primal > settings.tol_feas * (100.).as_T()
                        && self.res_primal > self.prev_res_primal * (100.).as_T())
                {
                    self.status = SolverStatus::InsufficientProgress;
                }
            }
        }

        // time or iteration limits
        // ----------------------
        if self.status == SolverStatus::Unsolved {
            if settings.max_iter == self.iterations {
                self.status = SolverStatus::MaxIterations;
            } else if self.solve_time > settings.time_limit {
                self.status = SolverStatus::MaxTime;
            }
        }

        // return TRUE if we settled on a final status
        self.status != SolverStatus::Unsolved
    }

    fn save_prev_iterate(&mut self, variables: &Self::V, prev_variables: &mut Self::V) {
        self.prev_cost_primal = self.cost_primal;
        self.prev_cost_dual = self.cost_dual;
        self.prev_res_primal = self.res_primal;
        self.prev_res_dual = self.res_dual;
        self.prev_gap_abs = self.gap_abs;
        self.prev_gap_rel = self.gap_rel;

        prev_variables.copy_from(variables);
    }

    fn reset_to_prev_iterate(&mut self, variables: &mut Self::V, prev_variables: &Self::V) {
        self.cost_primal = self.prev_cost_primal;
        self.cost_dual = self.prev_cost_dual;
        self.res_primal = self.prev_res_primal;
        self.res_dual = self.prev_res_dual;
        self.gap_abs = self.prev_gap_abs;
        self.gap_rel = self.prev_gap_rel;

        variables.copy_from(prev_variables);
    }

    fn save_best_iterate(
        &mut self,
        variables: &Self::V,
        best_variables: &mut Self::V,
        settings: &DefaultSettings<T>,
    ) {
        // Iterates on an infeasibility path (κ/τ > 1) are never candidates.
        if self.ktratio > T::one() {
            return;
        }

        // Strict improvement, tested so that a non-finite merit is rejected.  Every
        // comparison against NaN is false, so `merit >= best_merit` would treat a NaN
        // iterate as an improvement, overwrite a good one with it, and then disable the
        // fallback altogether, since the restore below requires a finite best_merit --
        // and a solve whose iterates have gone non-finite is exactly the case this is
        // meant to rescue.  The same test rejects an infinite merit, which is what a
        // zero reduced tolerance would produce.
        let merit = self.termination_merit(settings);
        if !(merit < self.best_merit) {
            return;
        }

        self.best_merit = merit;
        self.best_cost_primal = self.cost_primal;
        self.best_cost_dual = self.cost_dual;
        self.best_res_primal = self.res_primal;
        self.best_res_dual = self.res_dual;
        self.best_gap_abs = self.gap_abs;
        self.best_gap_rel = self.gap_rel;
        self.best_ktratio = self.ktratio;

        best_variables.copy_from(variables);
    }

    fn reset_to_best_iterate(
        &mut self,
        variables: &mut Self::V,
        best_variables: &Self::V,
        settings: &DefaultSettings<T>,
    ) {
        if !self.best_merit.is_finite() {
            return;
        }

        let merit = self.termination_merit(settings);
        if self.ktratio <= T::one() && merit <= self.best_merit {
            return;
        }

        self.cost_primal = self.best_cost_primal;
        self.cost_dual = self.best_cost_dual;
        self.res_primal = self.best_res_primal;
        self.res_dual = self.best_res_dual;
        self.gap_abs = self.best_gap_abs;
        self.gap_rel = self.best_gap_rel;
        self.ktratio = self.best_ktratio;

        variables.copy_from(best_variables);
    }

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32) {
        self.mu = μ;
        self.step_length = α;
        self.sigma = σ;
        self.iterations = iter;
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn set_status(&mut self, status: SolverStatus) {
        self.status = status;
    }
}

// Utility functions for convergence checkiing

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    fn check_convergence_full(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "full" tolerances
        let tol_gap_abs = settings.tol_gap_abs;
        let tol_gap_rel = settings.tol_gap_rel;
        let tol_feas = settings.tol_feas;
        let tol_infeas_abs = settings.tol_infeas_abs;
        let tol_infeas_rel = settings.tol_infeas_rel;
        let tol_ktratio = settings.tol_ktratio;

        let solved_status = SolverStatus::Solved;
        let pinf_status = SolverStatus::PrimalInfeasible;
        let dinf_status = SolverStatus::DualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    fn check_convergence_almost(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "almost" tolerances
        let tol_gap_abs = settings.reduced_tol_gap_abs;
        let tol_gap_rel = settings.reduced_tol_gap_rel;
        let tol_feas = settings.reduced_tol_feas;
        let tol_infeas_abs = settings.reduced_tol_infeas_abs;
        let tol_infeas_rel = settings.reduced_tol_infeas_rel;
        let tol_ktratio = settings.reduced_tol_ktratio;

        let solved_status = SolverStatus::AlmostSolved;
        let pinf_status = SolverStatus::AlmostPrimalInfeasible;
        let dinf_status = SolverStatus::AlmostDualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn check_convergence(
        &mut self,
        residuals: &DefaultResiduals<T>,
        tol_gap_abs: T,
        tol_gap_rel: T,
        tol_feas: T,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
        tol_ktratio: T,
        solved_status: SolverStatus,
        pinf_status: SolverStatus,
        dinf_status: SolverStatus,
    ) {
        if self.ktratio <= T::one() && self.is_solved(tol_gap_abs, tol_gap_rel, tol_feas) {
            self.status = solved_status;
        //PJG hardcoded factor 1000 here should be fixed
        } else if self.ktratio > tol_ktratio.recip() * (1000.0).as_T() {
            if self.is_primal_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = pinf_status;
            } else if self.is_dual_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = dinf_status;
            }
        }
    }

    // How far the current iterate is from satisfying `is_solved` at the *reduced*
    // tolerances: each quantity divided by the tolerance that will judge it, combined
    // exactly as in that test, so the duality gap enters through whichever of its
    // absolute/relative forms is closer to passing.  A merit below 1 passes.
    //
    // The reduced tolerances are the right yardstick because this merit only ever ranks
    // candidates for an unsuccessful exit, where `check_convergence_almost` -- which uses
    // exactly these tolerances -- decides whether the restored iterate can still be
    // reported as AlmostSolved.  Ranking by them makes the selection safe: the chosen
    // iterate has the smallest reduced merit of all candidates, so if any candidate would
    // have passed that check, the chosen one passes it too.  Comparing the raw quantities
    // instead can prefer an iterate that fails the check over one that passes, because the
    // tolerances differ from each other -- by default `reduced_tol_feas` is 1e-4 while
    // `reduced_tol_gap_rel` is 5e-5.
    fn termination_merit(&self, settings: &DefaultSettings<T>) -> T {
        let gap = T::min(
            self.gap_abs / settings.reduced_tol_gap_abs,
            self.gap_rel / settings.reduced_tol_gap_rel,
        );
        let feas = T::max(
            self.res_primal / settings.reduced_tol_feas,
            self.res_dual / settings.reduced_tol_feas,
        );
        T::max(gap, feas)
    }

    fn is_solved(&self, tol_gap_abs: T, tol_gap_rel: T, tol_feas: T) -> bool {
        ((self.gap_abs < tol_gap_abs) || (self.gap_rel < tol_gap_rel))
            && (self.res_primal < tol_feas)
            && (self.res_dual < tol_feas)
    }

    fn is_primal_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.dot_bz < -tol_infeas_abs)
            && (self.res_primal_inf < -tol_infeas_rel * residuals.dot_bz)
    }

    fn is_dual_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.dot_qx < -tol_infeas_abs)
            && (self.res_dual_inf < -tol_infeas_rel * residuals.dot_qx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::core::traits::Info;

    // fabricate an iterate whose merit-relevant fields are set directly, with x[0]
    // tagging which iterate it is so we can see which one survives
    fn set_iterate(
        info: &mut DefaultInfo<f64>,
        vars: &mut DefaultVariables<f64>,
        res: f64,
        ktratio: f64,
        tag: f64,
    ) {
        info.res_primal = res;
        info.res_dual = res;
        info.gap_rel = res;
        info.gap_abs = res;
        info.ktratio = ktratio;
        vars.x[0] = tag;
    }

    #[test]
    fn best_iterate_survives_a_degraded_final_iterate() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let mut vars = DefaultVariables::<f64>::new(1, 1);
        let mut best = DefaultVariables::<f64>::new(1, 1);
        info.reset(&mut Default::default());

        // a good iterate, then a better one, then the blowup that ends the solve
        set_iterate(&mut info, &mut vars, 1e-6, 0.5, 1.0);
        info.save_best_iterate(&vars, &mut best, &settings);
        set_iterate(&mut info, &mut vars, 1e-10, 0.5, 2.0);
        info.save_best_iterate(&vars, &mut best, &settings);
        set_iterate(&mut info, &mut vars, 3e-5, 0.5, 3.0);
        info.save_best_iterate(&vars, &mut best, &settings);

        info.reset_to_best_iterate(&mut vars, &best, &settings);

        assert_eq!(vars.x[0], 2.0, "the 1e-10 iterate should be restored");
        assert_eq!(info.res_primal, 1e-10);
        assert_eq!(info.gap_rel, 1e-10);
    }

    #[test]
    fn a_better_final_iterate_is_left_alone() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let mut vars = DefaultVariables::<f64>::new(1, 1);
        let mut best = DefaultVariables::<f64>::new(1, 1);
        info.reset(&mut Default::default());

        set_iterate(&mut info, &mut vars, 1e-6, 0.5, 1.0);
        info.save_best_iterate(&vars, &mut best, &settings);
        set_iterate(&mut info, &mut vars, 1e-9, 0.5, 2.0);

        info.reset_to_best_iterate(&mut vars, &best, &settings);

        assert_eq!(
            vars.x[0], 2.0,
            "the current iterate is the best one; keep it"
        );
        assert_eq!(info.res_primal, 1e-9);
    }

    // The acceptance check applied on an unsuccessful exit uses a different tolerance for
    // each quantity -- by default reduced_tol_feas is 1e-4 but reduced_tol_gap_rel is 5e-5 --
    // so ranking candidates by the raw worst quantity can prefer one that fails the check
    // over one that passes it.  Here the first candidate has the smaller raw maximum (8e-5
    // against 9e-5) yet its gap is outside tolerance, while the second is inside on every
    // count; the second must be the one restored.
    #[test]
    fn merit_ranks_by_distance_from_the_acceptance_check() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let mut vars = DefaultVariables::<f64>::new(1, 1);
        let mut best = DefaultVariables::<f64>::new(1, 1);
        info.reset(&mut Default::default());

        // candidate 1: gap outside the reduced tolerance, residuals well inside
        info.gap_abs = 8e-5;
        info.gap_rel = 8e-5;
        info.res_primal = 1e-5;
        info.res_dual = 1e-5;
        info.ktratio = 0.5;
        vars.x[0] = 1.0;
        info.save_best_iterate(&vars, &mut best, &settings);
        assert!(!info.is_solved(
            settings.reduced_tol_gap_abs,
            settings.reduced_tol_gap_rel,
            settings.reduced_tol_feas
        ));

        // candidate 2: larger raw maximum, but inside tolerance on every count
        info.gap_abs = 2e-5;
        info.gap_rel = 2e-5;
        info.res_primal = 9e-5;
        info.res_dual = 9e-5;
        vars.x[0] = 2.0;
        info.save_best_iterate(&vars, &mut best, &settings);
        assert!(info.is_solved(
            settings.reduced_tol_gap_abs,
            settings.reduced_tol_gap_rel,
            settings.reduced_tol_feas
        ));
        assert_eq!(
            best.x[0], 2.0,
            "the iterate that passes the check must be preferred"
        );

        // and it is the one a degraded final iterate is replaced by
        info.gap_abs = 1e-2;
        info.gap_rel = 1e-2;
        info.res_primal = 1e-2;
        info.res_dual = 1e-2;
        vars.x[0] = 3.0;
        info.reset_to_best_iterate(&mut vars, &best, &settings);
        assert_eq!(vars.x[0], 2.0);
        assert_eq!(info.res_primal, 9e-5);
    }

    // A solve whose iterates go non-finite is the case this fallback exists for, so a NaN
    // iterate must not be able to displace the good one that preceded it.  Comparisons
    // against NaN are all false, which makes this easy to get wrong.
    #[test]
    fn a_nan_iterate_cannot_displace_the_best_one() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let mut vars = DefaultVariables::<f64>::new(1, 1);
        let mut best = DefaultVariables::<f64>::new(1, 1);
        info.reset(&mut Default::default());

        set_iterate(&mut info, &mut vars, 1e-10, 0.5, 1.0);
        info.save_best_iterate(&vars, &mut best, &settings);

        // the solve blows up: residuals, gap and κ/τ are all NaN
        set_iterate(&mut info, &mut vars, f64::NAN, f64::NAN, 2.0);
        info.save_best_iterate(&vars, &mut best, &settings);
        assert_eq!(best.x[0], 1.0, "the NaN iterate must not be saved");
        assert!(
            info.best_merit.is_finite(),
            "the fallback must remain armed"
        );

        info.reset_to_best_iterate(&mut vars, &best, &settings);
        assert_eq!(vars.x[0], 1.0, "the good iterate must be restored");
        assert_eq!(info.res_primal, 1e-10);
    }

    #[test]
    fn iterates_on_the_infeasibility_path_are_not_candidates() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let mut vars = DefaultVariables::<f64>::new(1, 1);
        let mut best = DefaultVariables::<f64>::new(1, 1);
        info.reset(&mut Default::default());

        // tiny residuals but ktratio > 1: this is an infeasibility certificate path
        set_iterate(&mut info, &mut vars, 1e-12, 5.0, 1.0);
        info.save_best_iterate(&vars, &mut best, &settings);
        set_iterate(&mut info, &mut vars, 3e-5, 5.0, 2.0);
        info.reset_to_best_iterate(&mut vars, &best, &settings);

        assert_eq!(
            vars.x[0], 2.0,
            "nothing was ever saved, so nothing is restored"
        );
        assert!(!info.best_merit.is_finite());
    }
}
