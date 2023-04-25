#![allow(non_snake_case)]
use super::*;
use crate::algebra::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    traits::ProblemData,
};

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

    pub normq: T,
    pub normb: T,

    pub presolver: Presolver<T>,
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
        presolver: Presolver<T>,
    ) -> Self {
        // dimension checks will have already been
        // performed during problem setup, so skip here

        let P = P.to_triu();
        let q = q.to_vec();

        let (A, mut b) = {
            if let Some(map) = presolver.reduce_map.as_ref() {
                (
                    A.select_rows(&map.keep_logical),
                    b.select(&map.keep_logical),
                )
            } else {
                (A.clone(), b.to_vec())
            }
        };

        // cap entries in b at INFINITY.  This is important
        // for inf values that were not in a reduced cone
        let infbound = presolver.infbound.as_T();
        b.scalarop(|x| T::min(x, infbound));

        let (m, n) = A.size();

        let equilibration = DefaultEquilibrationData::<T>::new(n, m);

        let normq = q.norm_inf();
        let normb = b.norm_inf();

        Self {
            P,
            q,
            A,
            b,
            n,
            m,
            equilibration,
            normq,
            normb,
            presolver,
        }
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

            dwork.scalarop(|x| limit_scaling(x, scale_min, scale_max));
            ework.scalarop(|x| limit_scaling(x, scale_min, scale_max));

            dwork.rsqrt();
            ework.rsqrt();

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
                let scale_cost = limit_scaling(scale_cost, scale_min, scale_max);
                let ctmp = T::recip(scale_cost);

                // scale the penalty terms and overall scaling
                P.scale(ctmp);
                q.scale(ctmp);
                equil.c *= ctmp;
            }
        } //end Ruiz scaling loop

        // fix scalings in cones for which elementwise
        // scaling can't be applied
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

fn limit_scaling<T>(s: T, minval: T, maxval: T) -> T
where
    T: FloatT + ScalarMath<T = T>,
{
    s.clip(minval, maxval, T::one(), maxval)
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
            A.lrscale(e, d); // A[:,:] = Es*A*Ds
            q.hadamard(d);
        }
        None => {
            A.lscale(e); // A[:,:] = Es*A
        }
    }
    b.hadamard(e);
}
