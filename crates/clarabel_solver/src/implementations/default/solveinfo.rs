use super::*;
use crate::core::{
    Settings,
    cones::{CompositeCone,SupportedCones},
    components::{SolverStatus,SolveInfo},
};

use clarabel_algebra::*;
use clarabel_timers::Timers;
use std::time::Duration;

//PJG: do all of these need to be pub?  Used in finalization, for example

macro_rules! expformat {
    ($fmt:expr,$val:expr) => {
        _exp_str_reformat(format!($fmt, $val))
    };
}

#[derive(Default)]
pub struct DefaultSolveInfo<T> {
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

impl<T: FloatT> DefaultSolveInfo<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: FloatT> SolveInfo<T> for DefaultSolveInfo<T> {
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type R = DefaultResiduals<T>;
    type C = CompositeCone<T>;

    fn reset(&mut self) {
        self.status = SolverStatus::Unsolved;
        self.iterations = 0;
        self.solve_time = Duration::ZERO;
    }

    fn finalize(&mut self, timers: &Timers) {
        self.solve_time = timers.total_time();
    }

    fn print_header(
        &self,
        settings: &Settings<T>,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
    ) {
        if !settings.verbose {
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
        println!("problem: \n");
        println!("  variables     = {}", data.n);
        println!("  constraints   = {}", data.m);
        println!("  nnz(P)        = {}", data.P.nnz());
        println!("  nnz(A)        = {}", data.A.nnz());
        println!("  cones (total) = {}", cones.len());
        //PJG: Completing printing stuff here.  Additional functions needed
        _print_conedims_by_type(cones, SupportedCones::ZeroConeT(0));
        _print_conedims_by_type(cones, SupportedCones::NonnegativeConeT(0));
        _print_conedims_by_type(cones, SupportedCones::SecondOrderConeT(0));
        //_print_conedims_by_type(&cones, SupportedCones::PSDTriangleConeT); //PJG: SDP not implemented yet
        _print_settings(settings);
        println!();

        //print a subheader for the iterations info
        print!("iter    ");
        print!("pcost        ");
        print!("dcost       ");
        print!("pres      ");
        print!("dres      ");
        print!("k/t       ");
        print!(" μ       ");
        print!("step      ");
        println!();
        println!(
            "-----------------------------------------------------------------------------------"
        );
    }

    fn print_status(&self, settings: &Settings<T>) {
        if !settings.verbose {
            return;
        }

        print!("{:>3}  ", self.iterations);
        print!("{}  ", expformat!("{:+8.4e}", self.cost_primal));
        print!("{}  ", expformat!("{:+8.4e}", self.cost_dual));
        print!("{}  ", expformat!("{:6.2e}", self.res_primal));
        print!("{}  ", expformat!("{:6.2e}", self.res_dual));
        print!("{}  ", expformat!("{:6.2e}", self.ktratio));
        print!("{}  ", expformat!("{:6.2e}", self.μ));

        if self.iterations > 0 {
            print!("{}  ", expformat!("{:>.2e}", self.step_length));
        } else {
            print!(" ------   "); //info.step_length
        }

        println!();
    }

    fn print_footer(&self, settings: &Settings<T>) {
        if !settings.verbose {
            return;
        }

        println!(
            "-----------------------------------------------------------------------------------"
        );

        //PJG: no solver status string formatting available yet
        println!("Terminated with status = {}", self.status);

        println!("solve time = {:?}", self.solve_time);
    }

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
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
        //PJG: wtf is this?
        //self.get_solve_time();
    }

    fn check_termination(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &Settings<T>,
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
            } else if settings.time_limit > Duration::ZERO && self.solve_time > settings.time_limit
            {
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

fn _bool_on_off(v: bool) -> &'static str {
    match v {
        true => "on",
        false => "false",
    }
}

fn _print_settings<T: FloatT>(settings: &Settings<T>) {
    let set = settings;

    println!("settings:");

    if set.direct_kkt_solver {
        println!("  linear algebra: direct / TBD, precision: {} bit",  _get_precision_string::<T>());
        //set.direct_solve_method, _get_precision_string<T>());
    }

    let time_lim_str = {
        if set.time_limit == Duration::ZERO {
            "none".to_string()
        } else {
            format!("{:?}", set.time_limit)
        }
    };
    println!(
        "  max iter = {}, time limit = {:?},  max step = {:.3}",
        set.max_iter, time_lim_str, set.max_step_fraction
    );

    println!(
        "  tol_feas = {:.1e}, tol_abs = {:.1e}, tol_rel = {:.1e},",
        set.tol_feas, set.tol_gap_abs, set.tol_gap_rel
    );

    println!(
        "  static reg : {}, ϵ = {:.1e}",
        _bool_on_off(set.static_regularization_enable),
        set.static_regularization_eps
    );

    println!(
        "  dynamic reg: {}, ϵ = {:.1e}, δ = {:.1e}",
        _bool_on_off(set.dynamic_regularization_enable),
        set.dynamic_regularization_eps,
        set.dynamic_regularization_delta
    );

    println!(
        "  iter refine: {}, reltol = {:.1e}, abstol = {:.1e},",
        _bool_on_off(set.iterative_refinement_enable),
        set.iterative_refinement_reltol,
        set.iterative_refinement_abstol
    );

    println!(
        "               max iter = {}, stop ratio = {:.1}",
        set.iterative_refinement_max_iter, set.iterative_refinement_stop_ratio
    );

    println!(
        "  equilibrate: {}, min_scale = {:.1e}, max_scale = {:.1e}",
        _bool_on_off(set.equilibrate_enable),
        set.equilibrate_min_scaling,
        set.equilibrate_max_scaling
    );

    println!("               max iter = {}", set.equilibrate_max_iter,);
}

fn _get_precision_string<T: FloatT>() -> String {
    (::std::mem::size_of::<T>()*8).to_string()
}

//PJG: cone dimensions are now baked into SupportedCones<T>.  Maybe
//this function can be simplied.
fn _print_conedims_by_type<T: FloatT>(cones: &CompositeCone<T>, conetype: SupportedCones<T>) {
    let maxlistlen = 5;

    //skip if there are none of this type
    if !cones.type_counts.contains_key(&conetype.variant_name()) {
        return;
    }

    // how many of this type of cone?
    let name  = conetype.variant_name();
    let count = cones.type_counts[name];

    //let name  = rpad(string(type)[1:end-5],11)  #drops "ConeT part"
    let name = &name[0..name.len() - 5];
    let name = format!("{:>11}", name);

    let mut nvars = Vec::with_capacity(count);
    for (i, cone) in cones.iter().enumerate() {
        if cones.types[i] == conetype {
            nvars.push(cone.numel());
        }
    }
    print!("    : {} = {}, ", name, count);

    if count == 1 {
        print!(" numel = {}", nvars[0]);
    } else if count <= maxlistlen {
        //print them all
        print!(" numel = (");
        for nvar in nvars.iter().take(nvars.len() - 1) {
            print!("{},", nvar);
        }
        print!("{})", nvars[nvars.len() - 1]);
    } else {
        // print first (maxlistlen-1) and the final one
        print!(" numel = (");
        for nvar in nvars.iter().take(maxlistlen - 1) {
            print!("{},", nvar);
        }
        print!("...,{})", nvars[nvars.len() - 1]);
    }

    println!();
}

// convert a string in LowerExp display format into
// one that 1) always as a sign after the exponent,
// and 2) has at least two digits in the exponent.
// This matches the Julia output formatting.

fn _exp_str_reformat(mut thestr: String) -> String {
    // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
    let eidx = thestr.find('e').unwrap();
    let has_sign = thestr.chars().nth(eidx + 1).unwrap() == '-';

    let has_short_exp = {
        if !has_sign {
            thestr.len() == eidx + 2
        } else {
            thestr.len() == eidx + 3
        }
    };

    let chars;
    if !has_sign {
        if has_short_exp {
            chars = "+0";
        } else {
            chars = "+";
        }
    } else if has_short_exp {
        chars = "0";
    } else {
        chars = "";
    }

    let shift = if has_sign { 2 } else { 1 };
    thestr.insert_str(eidx + shift, chars);
    thestr
}
