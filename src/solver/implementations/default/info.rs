use super::*;
use crate::algebra::*;
use crate::solver::core::{traits::Info, SolverStatus};
use crate::timers::*;
use std::time::Duration;

#[derive(Default)]
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
    pub solve_time: Duration,
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
        self.solve_time = Duration::ZERO;

        timers.reset_timer("solve");
    }

    fn finalize(&mut self, timers: &mut Timers) {
        self.solve_time = timers.total_time();
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
        let two = T::from(2.).unwrap();
        let xPx_τinvsq_over2 = residuals.dot_xPx * τinv * τinv / two;
        self.cost_primal = (residuals.dot_qx * τinv + xPx_τinvsq_over2) / cscale;
        self.cost_dual = (-residuals.dot_bz * τinv - xPx_τinvsq_over2) / cscale;

        // primal and dual residuals.   Need to invert the equilibration
        self.res_primal = residuals.rz.norm_scaled(einv) * τinv;
        self.res_dual = residuals.rx.norm_scaled(dinv) * τinv;

        // primal and dual infeasibility residuals.   Need to invert the equilibration
        self.res_primal_inf = residuals.rx_inf.norm_scaled(dinv);
        self.res_dual_inf = T::max(
            residuals.Px.norm_scaled(dinv),
            residuals.rz_inf.norm_scaled(einv),
        );

        // absolute and relative gaps
        self.gap_abs = residuals.dot_sz * τinv * τinv;

        if (self.cost_primal > T::zero()) && (self.cost_dual < T::zero()) {
            self.gap_rel = T::max_value();
        } else {
            self.gap_rel = self.gap_abs / T::min(T::abs(self.cost_primal), T::abs(self.cost_dual));
        }

        // κ/τ
        self.ktratio = variables.κ / variables.τ;

        // solve time so far (includes setup)
        self.solve_time = timers.total_time();
    }

    fn check_termination(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        // optimality
        // ---------------------
        self.status = SolverStatus::Unsolved; //ensure default state

        if ((self.gap_abs < settings.tol_gap_abs) || (self.gap_rel < settings.tol_gap_rel))
            && (self.res_primal < settings.tol_feas)
            && (self.res_dual < settings.tol_feas)
        {
            self.status = SolverStatus::Solved;
        } else if self.ktratio > T::one() {
            if (residuals.dot_bz < -settings.tol_infeas_rel)
                && (self.res_primal_inf < -settings.tol_infeas_abs * residuals.dot_bz)
            {
                self.status = SolverStatus::PrimalInfeasible;
            } else if (residuals.dot_qx < -settings.tol_infeas_rel)
                && (self.res_dual_inf < -settings.tol_infeas_abs * residuals.dot_qx)
            {
                self.status = SolverStatus::DualInfeasible;
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

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32) {
        self.μ = μ;
        self.step_length = α;
        self.sigma = σ;
        self.iterations = iter;
    }
}
