#![allow(non_snake_case)]
use itertools::izip;

use super::*;
use crate::algebra::*;
use crate::solver::chordal::ChordalInfo;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    traits::ProblemData,
};
use crate::solver::SupportedConeT;

// convert every element of a tuple into an Option
macro_rules! into_options {
        () => { () };
        ($x:expr $(, $rest:expr)* ) => {
            (Some($x), $(Some($rest)),*)
        };
}

// ---------------
// Data type for default problem format
// ---------------

/// Standard-form solver type implementing the [`ProblemData`](crate::solver::core::traits::ProblemData) trait

pub struct DefaultProblemData<T> {
    // the main KKT residuals
    pub P: CscMatrix<T>,
    pub q: Vec<T>,
    pub A: CscMatrix<T>,
    pub b: Vec<T>,
    pub n: usize,
    pub m: usize,
    pub equilibration: DefaultEquilibrationData<T>,

    // unscaled inf norms of linear terms.  Set to "None"
    // during data updating to allow for multiple updates, and
    // then recalculated during solve if needed
    normq: Option<T>,
    normb: Option<T>,

    pub presolver: Option<Presolver<T>>,
    pub chordal_info: Option<ChordalInfo<T>>,
}

impl<T> DefaultProblemData<T>
where
    T: FloatT,
{
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: &DefaultSettings<T>,
    ) -> Self {
        // some caution is required to ensure we take a minimal,
        // but nonzero, number of data copies during presolve steps

        let mut P_new: Option<_> = None;
        let mut q_new: Option<_> = None;
        let mut A_new: Option<_> = None;
        let mut b_new: Option<_> = None;
        let mut cones_new: Option<_> = None;

        if !P.is_triu() {
            P_new = Some(P.to_triu());
            P = P_new.as_ref().unwrap();
        }

        // presolve : return nothing if disabled or no reduction
        // --------------------------------------
        let presolver = try_presolver(&A, &b, &cones, &settings);

        if let Some(presolver) = presolver {
            let (Some(A_new), Some(b_new), Some(cones_new)) = presolver.presolve(&A, &b, &cones);
            (A, b, cones) = (
                A_new.as_ref().unwrap(),
                b_new.as_ref().unwrap(),
                cones_new.as_ref().unwrap(),
            )
        }

        // chordal decomposition : return nothing if disabled or no decomp
        // --------------------------------------
        let chordal_info = try_chordal_info(&A, &b, &cones, &settings);

        if let Some(chordal_info) = chordal_info {
            let foo = into_options!(chordal_info.decomp_augment(&P, &q, &A, &b, &settings));
        }

        //  now make sure we have a clean copy of everything if we
        // haven't made one already.   Necessary since we will scale
        //  scale the internal copy and don't want to step on the user

        let P_new = P_new.unwrap_or_else(|| P.clone());
        let q_new = q_new.unwrap_or_else(|| q.to_vec());
        let A_new = A_new.unwrap_or_else(|| A.clone());
        let b_new = b_new.unwrap_or_else(|| b.to_vec());
        let cones_new = cones_new.unwrap_or_else(|| cones.clone());

        //cap entries in b at INFINITY.  This is important
        //for inf values that were not in a reduced cone
        //this is not considered part of the "presolve", so
        //can always happen regardless of user settings
        let infbound = presolver.infbound.as_T();
        b.scalarop(|x| T::min(x, infbound));

        // this ensures m is the *reduced* size m
        let (m, n) = A_new.size();

        let equilibration = DefaultEquilibrationData::<T>::new(n, m);

        let normq = Some(q_new.norm_inf());
        let normb = Some(b_new.norm_inf());

        Self {
            P: P_new,
            q: q_new,
            A: A_new,
            b: b_new,
            n,
            m,
            equilibration,
            normq,
            normb,
            presolver,
            chordal_info,
        }
    }

    pub(crate) fn get_normq(&mut self) -> T {
        if let Some(norm) = self.normq {
            norm
        } else {
            let dinv = &self.equilibration.dinv;
            let norm = self.q.norm_inf_scaled(dinv);
            self.normq = Some(norm);
            norm
        }
    }

    pub(crate) fn get_normb(&mut self) -> T {
        if let Some(norm) = self.normb {
            norm
        } else {
            let einv = &self.equilibration.einv;
            let norm = self.b.norm_inf_scaled(einv);
            self.normb = Some(norm);
            norm
        }
    }

    pub(crate) fn clear_normq(&mut self) {
        self.normq = None;
    }

    pub(crate) fn clear_normb(&mut self) {
        self.normb = None;
    }
}

impl<T> ProblemData<T> for DefaultProblemData<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn equilibrate(&mut self, cones: &CompositeCone<T>, settings: &DefaultSettings<T>) {
        let data = self;
        let equil = &mut data.equilibration;

        // if equilibration is disabled, just return.  Note that
        // the default equilibration structure initializes with
        // identity scaling already.
        if !settings.equilibrate_enable {
            return;
        }

        // references to scaling matrices from workspace
        let (d, e) = (&mut equil.d, &mut equil.e);

        // use the inverse scalings as work vectors
        let dwork = &mut equil.dinv;
        let ework = &mut equil.einv;

        // references to problem data
        // note that P may be triu, but it shouldn't matter
        let (P, A, q, b) = (&mut data.P, &mut data.A, &mut data.q, &mut data.b);

        let scale_min = settings.equilibrate_min_scaling;
        let scale_max = settings.equilibrate_max_scaling;

        // perform scaling operations for a fixed number of steps
        for _ in 0..settings.equilibrate_max_iter {
            kkt_col_norms(P, A, dwork, ework);

            //zero rows or columns should not get scaled
            dwork.scalarop(|x| if x == T::zero() { T::one() } else { x });
            ework.scalarop(|x| if x == T::zero() { T::one() } else { x });

            dwork.rsqrt();
            ework.rsqrt();

            // bound the cumulative scaling
            for (dwork, &d) in izip!(dwork.iter_mut(), d.iter()) {
                *dwork = T::clip(dwork, scale_min / d, scale_max / d);
            }
            for (ework, &e) in izip!(ework.iter_mut(), e.iter()) {
                *ework = T::clip(ework, scale_min / e, scale_max / e);
            }

            // Scale the problem data and update the
            // equilibration matrices
            scale_data(P, A, q, b, Some(dwork), ework);
            d.hadamard(dwork);
            e.hadamard(ework);

            // now use the Dwork array to hold the
            // column norms of the newly scaled P
            // so that we can compute the mean
            P.col_norms(dwork);
            let mean_col_norm_P = dwork.mean();
            let inf_norm_q = q.norm_inf();

            if mean_col_norm_P != T::zero() && inf_norm_q != T::zero() {
                let scale_cost = T::max(inf_norm_q, mean_col_norm_P);
                let ctmp = T::recip(scale_cost);
                let ctmp = T::clip(&ctmp, scale_min / equil.c, scale_max / equil.c);

                // scale the penalty terms and overall scaling
                P.scale(ctmp);
                q.scale(ctmp);
                equil.c *= ctmp;
            }
        } //end Ruiz scaling loop

        // fix scalings in cones for which elementwise
        // scaling can't be applied. Rectification should
        //either do nothing or take a convex combination of
        //scalings over a cone, so shouldn't need to check
        //bounds on the scalings here
        if cones.rectify_equilibration(ework, e) {
            // only rescale again if some cones were rectified
            scale_data(P, A, q, b, None, ework);
            e.hadamard(ework);
        }

        // update the inverse scaling data
        equil.dinv.scalarop_from(T::recip, d);
        equil.einv.scalarop_from(T::recip, e);
    }
}

// ---------------
// utilities
// ---------------

fn kkt_col_norms<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    norm_LHS: &mut [T],
    norm_RHS: &mut [T],
) {
    P.col_norms_sym(norm_LHS); // P can be triu
    A.col_norms_no_reset(norm_LHS); // incrementally from P norms
    A.row_norms(norm_RHS); // same as column norms of A'
}

fn scale_data<T: FloatT>(
    P: &mut CscMatrix<T>,
    A: &mut CscMatrix<T>,
    q: &mut [T],
    b: &mut [T],
    d: Option<&[T]>,
    e: &[T],
) {
    match d {
        Some(d) => {
            P.lrscale(d, d); // P[:,:] = Ds*P*Ds
            A.lrscale(e, d);
            q.hadamard(d);
        }
        None => {
            A.lscale(e); // A[:,:] = Es*A
        }
    }
    b.hadamard(e);
}

fn try_chordal_info<T>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Option<ChordalInfo<T>>
where
    T: FloatT,
{
    if !settings.chordal_decomposition_enable {
        return None;
    }

    // nothing to do if there are no PSD cones
    if !cones
        .iter()
        .any(|c| matches!(c, SupportedConeT::PSDTriangleConeT(_)))
    {
        return None;
    }

    let chordal_info = ChordalInfo::new(A, b, cones, settings);

    // no decomposition possible
    if !chordal_info.is_decomposed() {
        return None;
    }

    return Some(chordal_info);
}

fn try_presolver<T>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Option<Presolver<T>>
where
    T: FloatT,
{
    if !settings.presolve_enable {
        return None;
    }

    let presolver = Presolver::new(A, b, cones, settings);

    if !presolver.is_reduced() {
        return None;
    }

    return Some(presolver);
}
