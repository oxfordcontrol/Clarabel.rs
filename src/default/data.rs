#![allow(non_snake_case)]
use crate::algebra::*;
use crate::default::equilibration::DefaultEquilibration;
use crate::ConeSet;
use crate::cones::Cone;
use crate::ProblemData;
use crate::settings::Settings;

// ---------------
// Data type for default problem format
// ---------------

pub struct DefaultProblemData<T: FloatT = f64> {

    // the main KKT residuals
    P: CscMatrix<T>,
    q: Vec<T>,
    A: CscMatrix<T>,
    b: Vec<T>,
    n: usize,
    m: usize,
    equilibration: DefaultEquilibration<T>,

}

impl<T: FloatT> DefaultProblemData<T> {
    pub fn new(
        P: CscMatrix<T>,
        q: Vec<T>,
        A: CscMatrix<T>,
        b: Vec<T>,
        cones:ConeSet<T>
    ) -> Self
{
        let (m,n) = (b.len(),q.len());

        assert_eq!(m,A.nrows());
        assert_eq!(n,A.ncols());
        assert_eq!(n,P.ncols());
        assert!(P.is_square());

        let P = P.clone();
        let q = q.clone();
        let A = A.clone();
        let b = b.clone();
        let equilibration = DefaultEquilibration::<T>::new(n,cones);

        Self {P: P, q: q, A: A, b:b, n: n, m: m, equilibration: equilibration}
    }
}


impl<T> ProblemData<T> for DefaultProblemData<T>
where
    T: FloatT,
{

    fn equilibrate(&mut self, cones: &ConeSet<T>, settings: &Settings){

        let data = self;
        let equil = &mut data.equilibration;

        // if equilibration is disabled, just return.  Note that
        // the default equilibration structure initializes with
        // identity scaling already.
        if !settings.equilibrate_enable {
            return
        }

        // references to scaling matrices from workspace
        let (d, e) = (&mut equil.d, &mut equil.e);

        // use the inverse scalings as work vectors
        let dwork = &mut equil.dinv;
        let ework = &mut equil.einv;

        // references to problem data
        // note that P may be triu, but it shouldn't matter
        let (P,A,q,b) = (&mut data.P,&mut data.A,&mut data.q,&mut data.b);

        let scale_min = settings.equilibrate_min_scaling;
        let scale_max = settings.equilibrate_max_scaling;

        // perform scaling operations for a fixed number of steps
        for i in 0..settings.equilibrate_max_iter {
            //
            // 	kkt_col_norms!(P, A, dwork, ework)
            //
            // 	limit_scaling!(dwork, scale_min, scale_max)
            // 	limit_scaling!(ework, scale_min, scale_max)
            //
            dwork.rsqrt();
            ework.rsqrt();

            // Scale the problem data and update the
            // equilibration matrices
            //scale_data!(P, A, q, b, dwork, ework, T::one());
            d.hadamard(dwork);
            e.hadamard(ework);

            // now use the Dwork array to hold the
            // column norms of the newly scaled P
            // so that we can compute the mean
            P.col_norms(dwork);
            let mean_col_norm_P = dwork.mean();
            let inf_norm_q      = q.norm_inf();

            if mean_col_norm_P  != T::zero() && inf_norm_q != T::zero(){

                let scale_cost = T::max(inf_norm_q, mean_col_norm_P);
                //let scale_cost = limit_scaling(scale_cost, scale_min, scale_max);
                let ctmp = T::recip(scale_cost);

                // scale the penalty terms and overall scaling
                P.scale(ctmp);
                q.scale(ctmp);
                equil.c *= ctmp;
            }

        } //end Ruiz scaling loop

    	// fix scalings in cones for which elementwise
        // scaling can't be applied
    	if cones.rectify_equilibration(ework, e){
    		// only rescale again of some cones were rectified
    		// scale_data!(P, A, q, b, nothing, ework, one(T))
    		e.hadamard(ework);
    	}

    	// update the inverse scaling data
        equil.dinv.copy_from(d);
        equil.einv.copy_from(e);
        equil.dinv.reciprocal();
        equil.einv.reciprocal();
    }
}
