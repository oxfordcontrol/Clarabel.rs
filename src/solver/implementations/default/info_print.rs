#[cfg(feature = "sdp")]
use crate::solver::chordal::ChordalInfo;

use crate::stdio;
use crate::{
    algebra::*,
    solver::core::cones::{SupportedConeAsTag, SupportedConeTag},
};
use std::io::Write;

use super::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    traits::InfoPrint,
};
use std::time::Duration;

macro_rules! expformat {
    ($fmt:expr,$val:expr) => {
        if $val.is_finite() {
            _exp_str_reformat(format!($fmt, $val))
        } else {
            format!($fmt, $val)
        }
    };
}

impl<T> InfoPrint<T> for DefaultInfo<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn print_configuration(
        &self,
        settings: &DefaultSettings<T>,
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
    ) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let mut out = stdio::stdout();

        if let Some(ref presolver) = data.presolver {
            writeln!(
                out,
                "\npresolve: removed {} constraints",
                presolver.count_reduced()
            )?;
        }

        #[cfg(feature = "sdp")]
        if let Some(ref chordal_info) = data.chordal_info {
            print_chordal_decomposition(chordal_info, settings)?;
        }

        writeln!(out, "\nproblem:")?;
        writeln!(out, "  variables     = {}", data.n)?;
        writeln!(out, "  constraints   = {}", data.m)?;
        writeln!(out, "  nnz(P)        = {}", data.P.nnz())?;
        writeln!(out, "  nnz(A)        = {}", data.A.nnz())?;
        writeln!(out, "  cones (total) = {}", cones.len())?;

        //All dims here are dummies since we just care about the cone type
        _print_conedims_by_type(cones, SupportedConeTag::ZeroCone)?;
        _print_conedims_by_type(cones, SupportedConeTag::NonnegativeCone)?;
        _print_conedims_by_type(cones, SupportedConeTag::SecondOrderCone)?;
        _print_conedims_by_type(cones, SupportedConeTag::ExponentialCone)?;
        _print_conedims_by_type(cones, SupportedConeTag::PowerCone)?;
        _print_conedims_by_type(cones, SupportedConeTag::GenPowerCone)?;
        #[cfg(feature = "sdp")]
        _print_conedims_by_type(cones, SupportedConeTag::PSDTriangleCone)?;

        writeln!(out,)?;
        _print_settings(settings)?;
        writeln!(out,)?;

        std::io::Result::Ok(())
    }

    fn print_status_header(&self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let mut out = stdio::stdout();

        //print a subheader for the iterations info
        write!(out, "iter    ")?;
        write!(out, "pcost        ")?;
        write!(out, "dcost       ")?;
        write!(out, "gap       ")?;
        write!(out, "pres      ")?;
        write!(out, "dres      ")?;
        write!(out, "k/t       ")?;
        write!(out, " μ       ")?;
        write!(out, "step      ")?;
        writeln!(out,)?;
        writeln!(out,
            "---------------------------------------------------------------------------------------------"
        )?;
        stdio::stdout().flush()?;
        std::io::Result::Ok(())
    }

    fn print_status(&self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let mut out = stdio::stdout();

        write!(out, "{:>3}  ", self.iterations)?;
        write!(out, "{}  ", expformat!("{:+8.4e}", self.cost_primal))?;
        write!(out, "{}  ", expformat!("{:+8.4e}", self.cost_dual))?;
        let gapprint = T::min(self.gap_abs, self.gap_rel);
        write!(out, "{}  ", expformat!("{:6.2e}", gapprint))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.res_primal))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.res_dual))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.ktratio))?;
        write!(out, "{}  ", expformat!("{:6.2e}", self.μ))?;

        if self.iterations > 0 {
            write!(out, "{}  ", expformat!("{:>.2e}", self.step_length))?;
        } else {
            write!(out, " ------   ")?; //info.step_length
        }

        writeln!(out,)?;

        std::io::Result::Ok(())
    }

    fn print_footer(&self, settings: &DefaultSettings<T>) -> std::io::Result<()> {
        if !settings.verbose {
            return std::io::Result::Ok(());
        }

        let mut out = stdio::stdout();

        writeln!(out,
            "---------------------------------------------------------------------------------------------"
        )?;

        writeln!(out, "Terminated with status = {}", self.status)?;

        writeln!(
            out,
            "solve time = {:?}",
            Duration::from_secs_f64(self.solve_time)
        )?;

        std::io::Result::Ok(())
    }
}

fn _bool_on_off(v: bool) -> &'static str {
    match v {
        true => "on",
        false => "false",
    }
}

fn _print_settings<T: FloatT>(settings: &DefaultSettings<T>) -> std::io::Result<()> {
    let set = settings;
    let mut out = stdio::stdout();

    writeln!(out, "settings:")?;

    if set.direct_kkt_solver {
        write!(
            out,
            "  linear algebra: direct / {}, precision: {} bit",
            set.direct_solve_method,
            _get_precision_string::<T>()
        )?;
        print_nthreads(&mut out, settings)?;
        writeln!(out)?;
    }

    let time_lim_str = {
        if set.time_limit.is_infinite() {
            "Inf".to_string()
        } else {
            format!("{:?}", set.time_limit)
        }
    };
    writeln!(
        out,
        "  max iter = {}, time limit = {},  max step = {:.3}",
        set.max_iter, time_lim_str, set.max_step_fraction
    )?;

    writeln!(
        out,
        "  tol_feas = {:.1e}, tol_gap_abs = {:.1e}, tol_gap_rel = {:.1e},",
        set.tol_feas, set.tol_gap_abs, set.tol_gap_rel
    )?;

    writeln!(
        out,
        "  static reg : {}, ϵ1 = {:.1e}, ϵ2 = {:.1e}",
        _bool_on_off(set.static_regularization_enable),
        set.static_regularization_constant,
        set.static_regularization_proportional,
    )?;

    writeln!(
        out,
        "  dynamic reg: {}, ϵ = {:.1e}, δ = {:.1e}",
        _bool_on_off(set.dynamic_regularization_enable),
        set.dynamic_regularization_eps,
        set.dynamic_regularization_delta
    )?;

    writeln!(
        out,
        "  iter refine: {}, reltol = {:.1e}, abstol = {:.1e},",
        _bool_on_off(set.iterative_refinement_enable),
        set.iterative_refinement_reltol,
        set.iterative_refinement_abstol
    )?;

    writeln!(
        out,
        "               max iter = {}, stop ratio = {:.1}",
        set.iterative_refinement_max_iter, set.iterative_refinement_stop_ratio
    )?;

    writeln!(
        out,
        "  equilibrate: {}, min_scale = {:.1e}, max_scale = {:.1e}",
        _bool_on_off(set.equilibrate_enable),
        set.equilibrate_min_scaling,
        set.equilibrate_max_scaling
    )?;

    writeln!(
        out,
        "               max iter = {}",
        set.equilibrate_max_iter,
    )?;

    std::io::Result::Ok(())
}

#[allow(unused_variables)] //out is unused if faer-sparse is not enabled
fn print_nthreads<T: FloatT>(
    out: &mut stdio::Stdout,
    settings: &DefaultSettings<T>,
) -> std::io::Result<()> {
    match settings.direct_solve_method.as_str() {
        #[cfg(feature = "faer-sparse")]
        "faer" => {
            let nthreads =
                crate::solver::core::kktsolvers::direct::ldlsolvers::faer_ldl::FaerDirectLDLSolver::<
                    T,
                >::nthreads_from_settings(settings.max_threads as usize);
            if nthreads == 1 {
                write!(out, " ({nthreads} thread) ")
            } else {
                write!(out, " ({nthreads} threads) ")
            }
        }
        _ => std::io::Result::Ok(()),
    }
}

#[cfg(feature = "sdp")]
fn print_chordal_decomposition<T: FloatT>(
    chordal_info: &ChordalInfo<T>,
    settings: &DefaultSettings<T>,
) -> std::io::Result<()> {
    let mut out = stdio::stdout();

    writeln!(out, "\nchordal decomposition:")?;
    writeln!(
        out,
        "  compact format = {}, dual completion = {}",
        _bool_on_off(settings.chordal_decomposition_compact),
        _bool_on_off(settings.chordal_decomposition_complete_dual)
    )?;

    writeln!(
        out,
        "  merge method = {}",
        settings.chordal_decomposition_merge_method
    )?;

    writeln!(
        out,
        "  PSD cones initial             = {}",
        chordal_info.init_psd_cone_count()
    )?;

    writeln!(
        out,
        "  PSD cones decomposable        = {}",
        chordal_info.decomposable_cone_count()
    )?;

    writeln!(
        out,
        "  PSD cones after decomposition = {}",
        chordal_info.premerge_psd_cone_count()
    )?;

    writeln!(
        out,
        "  PSD cones after merges        = {}",
        chordal_info.final_psd_cone_count()
    )?;

    std::io::Result::Ok(())
}

fn _get_precision_string<T: FloatT>() -> String {
    (::std::mem::size_of::<T>() * 8).to_string()
}

fn _print_conedims_by_type<T: FloatT>(
    cones: &CompositeCone<T>,
    conetag: SupportedConeTag,
) -> std::io::Result<()> {
    let maxlistlen = 5;

    let count = cones.get_type_count(conetag);

    //skip if there are none of this type
    if count == 0 {
        return std::io::Result::Ok(());
    }

    let mut out = stdio::stdout();

    // how many of this type of cone?
    let name = conetag.as_str();

    // drops trailing "Cone" part of name
    let name = &name[0..name.len() - 4];
    let name = format!("{:>11}", name);

    let mut nvars = Vec::with_capacity(count);
    for cone in cones.iter() {
        if cone.as_tag() == conetag {
            nvars.push(cone.numel());
        }
    }
    write!(out, "    : {} = {}, ", name, count)?;

    if count == 1 {
        write!(out, " numel = {}", nvars[0])?;
    } else if count <= maxlistlen {
        //print them all
        write!(out, " numel = (")?;
        for nvar in nvars.iter().take(nvars.len() - 1) {
            write!(out, "{},", nvar)?;
        }
        write!(out, "{})", nvars[nvars.len() - 1])?;
    } else {
        // print first (maxlistlen-1) and the final one
        write!(out, " numel = (")?;
        for nvar in nvars.iter().take(maxlistlen - 1) {
            write!(out, "{},", nvar)?;
        }
        write!(out, "...,{})", nvars[nvars.len() - 1])?;
    }

    writeln!(out,)?;

    std::io::Result::Ok(())
}

// convert a string in LowerExp display format into one that
// 1) always has a sign after the exponent, and
// 2) has at least two digits in the exponent.
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
