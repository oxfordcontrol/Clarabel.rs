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
    /// solve time
    pub solve_time: f64,
    /// solver status
    pub status: SolverStatus,

    /// linear solver information
    pub linsolver: LinearSolverInfo,

    /// Per-iteration trace of (iter, max_numer_bits, max_denom_bits)
    /// across the primal `x` vector. Populated only on backends with
    /// arbitrary-precision representations (e.g. `RationalReal`,
    /// `MpfrFloat`); for IEEE floats this stays empty (the underlying
    /// [`BitWidthDiagnostic`](crate::algebra::BitWidthDiagnostic)
    /// returns `(0, 0)` and we skip the push). Useful for observing
    /// rational denominator blow-up across an IPM run; QOU uses this
    /// to predict whether a problem at H_n is tractable a priori.
    pub iter_diagnostics: Vec<IterDiagnostic>,

    // target stream for printing
    pub(crate) stream: PrintTarget,
}

/// One row of the per-iteration diagnostic trace.
/// See [`DefaultInfo::iter_diagnostics`].
#[derive(Default, Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IterDiagnostic {
    /// Iteration number at which this row was recorded (matches
    /// `info.iterations` at the moment of capture).
    pub iter: u32,
    /// Maximum numerator bit-length across the primal `x` vector.
    pub max_numer_bits: u64,
    /// Maximum denominator bit-length across the primal `x` vector.
    pub max_denom_bits: u64,
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
        let τinv = T::recip(variables.τ.clone());

        // unscaled linear term norms
        let normb = data.get_normb();
        let normq = data.get_normq();

        // shortcuts for the equilibration matrices
        let d = &data.equilibration.d;
        let e = &data.equilibration.e;
        let dinv = &data.equilibration.dinv;
        let einv = &data.equilibration.einv;
        let cinv = T::recip(data.equilibration.c.clone());

        // primal and dual costs. dot products are invariant w.r.t
        // equilibration, but we still need to back out the overall
        // objective scaling term c

        let xPx_τinvsq_over2 =
            residuals.dot_xPx.clone() * τinv.clone() * τinv.clone() / (2.).as_T();
        self.cost_primal =
            (residuals.dot_qx.clone() * τinv.clone() + xPx_τinvsq_over2.clone()) * cinv.clone();
        self.cost_dual =
            (-residuals.dot_bz.clone() * τinv.clone() - xPx_τinvsq_over2) * cinv.clone();

        // variables norms, undoing the equilibration.  Do not unscale
        // by τ yet because the infeasibility residuals are ratios of
        // terms that have no affine parts anyway
        let mut normx = variables.x.norm_scaled(d);
        let mut normz = variables.z.norm_scaled(e) * cinv.clone();
        let mut norms = variables.s.norm_scaled(einv);

        // primal and dual infeasibility residuals.
        self.res_primal_inf =
            (residuals.rx_inf.norm_scaled(dinv) * cinv.clone()) / T::max(T::one(), normz.clone());
        self.res_dual_inf = T::max(
            residuals.Px.norm_scaled(dinv) / T::max(T::one(), normx.clone()),
            residuals.rz_inf.norm_scaled(einv) / T::max(T::one(), normx.clone() + norms.clone()),
        );

        // now back out the τ scaling so we can normalize the unscaled primal / dual errors
        normx *= τinv.clone();
        normz *= τinv.clone();
        norms *= τinv.clone();

        // primal and dual relative residuals.
        self.res_primal = residuals.rz.norm_scaled(einv) * τinv.clone()
            / T::max(T::one(), normb + normx.clone() + norms);
        self.res_dual = residuals.rx.norm_scaled(dinv) * τinv.clone() * cinv
            / T::max(T::one(), normq + normx + normz);

        // absolute and relative gaps
        self.gap_abs = (self.cost_primal.clone() - self.cost_dual.clone()).abs();
        self.gap_rel = self.gap_abs.clone()
            / T::max(
                T::one(),
                T::min(self.cost_primal.clone().abs(), self.cost_dual.clone().abs()),
            );

        // κ/τ ratio (scaled)
        self.ktratio = variables.κ.clone() * τinv;

        // solve time so far (includes setup)
        self.solve_time = timers.total_time().as_secs_f64();

        // Per-iteration bit-width trace (only populated when T's
        // BitWidthDiagnostic returns non-zero values, i.e. on
        // arbitrary-precision backends like RationalReal). Auto-import
        // via the FloatT supertrait chain — no explicit `use` needed.
        use crate::algebra::BitWidthDiagnostic as _;
        let (n_max, d_max) = variables
            .x
            .iter()
            .map(|xi| xi.bit_width())
            .fold((0u64, 0u64), |a, b| (a.0.max(b.0), a.1.max(b.1)));
        if n_max != 0 || d_max != 0 {
            self.iter_diagnostics.push(IterDiagnostic {
                iter: self.iterations,
                max_numer_bits: n_max,
                max_denom_bits: d_max,
            });
        }
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
                if (self.res_dual > settings.tol_feas.clone() * (100.).as_T()
                    && self.res_dual > self.prev_res_dual.clone() * (100.).as_T())
                    || (self.res_primal > settings.tol_feas.clone() * (100.).as_T()
                        && self.res_primal > self.prev_res_primal.clone() * (100.).as_T())
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
        self.prev_cost_primal = self.cost_primal.clone();
        self.prev_cost_dual = self.cost_dual.clone();
        self.prev_res_primal = self.res_primal.clone();
        self.prev_res_dual = self.res_dual.clone();
        self.prev_gap_abs = self.gap_abs.clone();
        self.prev_gap_rel = self.gap_rel.clone();

        prev_variables.copy_from(variables);
    }

    fn reset_to_prev_iterate(&mut self, variables: &mut Self::V, prev_variables: &Self::V) {
        self.cost_primal = self.prev_cost_primal.clone();
        self.cost_dual = self.prev_cost_dual.clone();
        self.res_primal = self.prev_res_primal.clone();
        self.res_dual = self.prev_res_dual.clone();
        self.gap_abs = self.prev_gap_abs.clone();
        self.gap_rel = self.prev_gap_rel.clone();

        variables.copy_from(prev_variables);
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
        let tol_gap_abs = settings.tol_gap_abs.clone();
        let tol_gap_rel = settings.tol_gap_rel.clone();
        let tol_feas = settings.tol_feas.clone();
        let tol_infeas_abs = settings.tol_infeas_abs.clone();
        let tol_infeas_rel = settings.tol_infeas_rel.clone();
        let tol_ktratio = settings.tol_ktratio.clone();

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
        let tol_gap_abs = settings.reduced_tol_gap_abs.clone();
        let tol_gap_rel = settings.reduced_tol_gap_rel.clone();
        let tol_feas = settings.reduced_tol_feas.clone();
        let tol_infeas_abs = settings.reduced_tol_infeas_abs.clone();
        let tol_infeas_rel = settings.reduced_tol_infeas_rel.clone();
        let tol_ktratio = settings.reduced_tol_ktratio.clone();

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
            if self.is_primal_infeasible(residuals, tol_infeas_abs.clone(), tol_infeas_rel.clone())
            {
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
            && (self.res_primal_inf < -tol_infeas_rel * residuals.dot_bz.clone())
    }

    fn is_dual_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.dot_qx < -tol_infeas_abs)
            && (self.res_dual_inf < -tol_infeas_rel * residuals.dot_qx.clone())
    }
}
