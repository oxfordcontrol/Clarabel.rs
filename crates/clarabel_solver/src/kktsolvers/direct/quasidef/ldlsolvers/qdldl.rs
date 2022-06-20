#![allow(non_snake_case)]
use crate::kktsolvers::direct::DirectLDLSolver;
use crate::Settings;
use clarabel_algebra::*;
use clarabel_qdldl::*;
//PJG: includes WTF

pub struct QDLDLDirectLDLSolver<T: FloatT> {
    //KKT matrix and its QDLDL factorization
    KKT: CscMatrix<T>,
    factors: QDLDLFactorisation<T>,

    //internal workspace for IR scheme.  This also
    //requires that we carry a separate unpermuted
    //copy of Dsigns here.
    //PJG: This should be dropped and IR implemented
    //internally in the QDLDL crate
    Dsigns: Vec<i8>,
    work: Vec<T>,
}

impl<T: FloatT> QDLDLDirectLDLSolver<T> {
    pub fn new(KKT: CscMatrix<T>, Dsigns: Vec<i8>, settings: &Settings<T>) -> Self {
        let dim = KKT.nrows();

        assert!(dim == KKT.ncols(), "KKT matrix is not square");

        //construct the LDL solver settings
        let opts = QDLDLSettingsBuilder::default()
            .logical(true) //allocate memory only on init
            .Dsigns(Dsigns.to_vec())
            .regularize_enable(true)
            .regularize_eps(settings.dynamic_regularization_eps)
            .regularize_delta(settings.dynamic_regularization_delta)
            .build()
            .unwrap();

        let factors = QDLDLFactorisation::<T>::new(&KKT, Some(opts));

        let work = vec![T::zero(); dim];

        Self {
            KKT,
            factors,
            Dsigns,
            work,
        }
    }
}

impl<T: FloatT> DirectLDLSolver<T> for QDLDLDirectLDLSolver<T> {
    fn update_values(&mut self, index: &[usize], values: &[T]) {
        // Updating values in both the KKT matrix and
        // in the reordered copy held internally by QDLDL.
        // The former is needed for iterative refinement since
        // QDLDL does not have internal iterative refinement

        self.factors.update_values(index, values);

        for (idx, v) in index.iter().zip(values.iter()) {
            self.KKT.nzval[*idx] = *v;
        }
    }

    fn scale_values(&mut self, index: &[usize], scale: T) {
        // Scale values in both the KKT matrix and
        // in the reordered copy held internally by QDLDL.
        // The former is needed for iterative refinement since
        // QDLDL does not have internal iterative refinement

        self.factors.scale_values(index, scale);

        for idx in index.iter() {
            self.KKT.nzval[*idx] *= scale;
        }
    }

    fn offset_values(&mut self, index: &[usize], offset: T) {
        self.factors.offset_values(index, offset);

        for (idx, sign) in index.iter().zip(self.Dsigns.iter()) {
            let sign = T::from_i8(*sign).unwrap();
            self.KKT.nzval[*idx] += sign * offset;
        }
    }

    fn solve(&mut self, x: &mut [T], b: &[T], settings: &Settings<T>) {
        // make an initial solve (solves in place)
        x.copy_from(b);
        self.factors.solve(x);

        if settings.iterative_refinement_enable {
            self.iterative_refinement(x, b, settings);
        }
    }

    fn refactor(&mut self) {
        self.factors.refactor();
    }
}

impl<T: FloatT> QDLDLDirectLDLSolver<T> {
    fn iterative_refinement(&mut self, x: &mut [T], b: &[T], settings: &Settings<T>) {
        let work = &mut self.work;

        // iterative refinement params
        let reltol = settings.iterative_refinement_reltol;
        let abstol = settings.iterative_refinement_abstol;
        let maxiter = settings.iterative_refinement_max_iter;
        let stopratio = settings.iterative_refinement_stop_ratio;

        // Note that K is only triu data, so need to
        // be careful when computing the residual here
        let K = &self.KKT;

        let lastnorme = T::infinity();

        let normb = b.norm_inf();

        for _ in 0..maxiter {
            // this is work = error = b - Kξ
            work.copy_from(b);
            K.symv(work, MatrixTriangle::Triu, x, -T::one(), T::one());
            let norme = work.norm_inf();

            // test for convergence before committing
            // to a refinement step
            if norme <= (abstol + reltol * normb) {
                break;
            }

            // if we haven't improved by at least the halting
            // ratio since the last pass through, then abort
            if lastnorme / norme < stopratio {
                break;
            }

            // make a refinement and continue
            self.factors.solve(work); //this is Δξ
            x.axpby(T::one(), work, T::one()); //x .+= work
        }
    }
}
