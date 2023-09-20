use super::*;
use crate::algebra::*;
use crate::solver::core::{traits::Info, SolverStatus};
use crate::solver::traits::Variables;
use crate::timers::*;

/// Standard-form solver type implementing the [`Info`](crate::solver::core::traits::Info) and [`InfoPrint`](crate::solver::core::traits::InfoPrint) traits

#[repr(C)]
#[derive(Default, Debug, Clone)]
pub struct DefaultInfo<T> {
    pub μ: T,
    pub sigma: T,
    pub step_length: T,
    pub iterations: u32,
    pub cost_primal: T,
    pub cost_dual: T,
    pub res_primal: T,
    pub res_dual: T,
    pub res_primal_inf: T,
    pub res_dual_inf: T,
    pub gap_abs: T,
    pub gap_rel: T,
    pub ktratio: T,

    // previous iterate
    prev_cost_primal: T,
    prev_cost_dual: T,
    prev_res_primal: T,
    prev_res_dual: T,
    prev_gap_abs: T,
    prev_gap_rel: T,

    pub solve_time: f64,
    pub status: SolverStatus,
}

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    pub fn new() -> Self {
        Self::default()
    }
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

        timers.reset_timer("solve");
    }

    fn finalize(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
        timers: &mut Timers,
    ) {
        // if there was an error or we ran out of time
        // or iterations, check for partial convergence

        if self.status.is_errored()
            || matches!(self.status, SolverStatus::MaxIterations)
            || matches!(self.status, SolverStatus::MaxTime)
        {
            self.check_convergence_almost(residuals, settings);
        }

        self.solve_time = timers.total_time().as_secs_f64();
    }

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        timers: &Timers,
    ) {
        // optimality termination check should be computed w.r.t
        // the pre-homogenization x and z variables.
        let τinv = T::recip(variables.τ);

        // shortcuts for the equilibration matrices
        let dinv = &data.equilibration.dinv;
        let einv = &data.equilibration.einv;
        let cscale = data.equilibration.c;

        // primal and dual costs. dot products are invariant w.r.t
        // equilibration, but we still need to back out the overall
        // objective scaling term c

        let xPx_τinvsq_over2 = residuals.dot_xPx * τinv * τinv / (2.).as_T();
        self.cost_primal = (residuals.dot_qx * τinv + xPx_τinvsq_over2) / cscale;
        self.cost_dual = (-residuals.dot_bz * τinv - xPx_τinvsq_over2) / cscale;

        //primal and dual relative residuals.   Need to invert the equilibration
        let normx = variables.x.norm_scaled(dinv) * τinv;
        let normz = variables.z.norm_scaled(einv) * τinv;
        let norms = variables.s.norm_scaled(einv) * τinv;

        // primal and dual residuals.   Need to invert the equilibration
        self.res_primal =
            residuals.rz.norm_scaled(einv) * τinv / T::max(T::one(), data.normb + normx + norms);
        self.res_dual =
            residuals.rx.norm_scaled(dinv) * τinv / T::max(T::one(), data.normq + normx + normz);

        // primal and dual infeasibility residuals.   Need to invert the equilibration
        self.res_primal_inf = residuals.rx_inf.norm_scaled(dinv) / T::max(T::one(), normz);
        self.res_dual_inf = T::max(
            residuals.Px.norm_scaled(dinv) / T::max(T::one(), normx),
            residuals.rz_inf.norm_scaled(einv) / T::max(T::one(), normx + norms),
        );

        // absolute and relative gaps
        self.gap_abs = T::abs(self.cost_primal - self.cost_dual);
        self.gap_rel = self.gap_abs
            / T::max(
                T::one(),
                T::min(T::abs(self.cost_primal), T::abs(self.cost_dual)),
            );

        // κ/τ
        self.ktratio = variables.κ / variables.τ;

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
            if (self.res_dual > settings.tol_feas
                && self.res_dual > self.prev_res_dual * (100.).as_T())
                || (self.res_primal > settings.tol_feas
                    && self.res_primal > self.prev_res_primal * (100.).as_T())
            {
                self.status = SolverStatus::InsufficientProgress;
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

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32) {
        self.μ = μ;
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
