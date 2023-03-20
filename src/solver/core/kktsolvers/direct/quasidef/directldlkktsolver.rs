#![allow(non_snake_case)]

use super::ldlsolvers::qdldl::*;
use super::*;
use crate::solver::core::kktsolvers::KKTSolver;
use crate::solver::core::{cones::*, CoreSettings};
use std::iter::zip;

// -------------------------------------
// KKTSolver using direct LDL factorisation
// -------------------------------------

// We require Send here to allow pyo3 builds to share
// solver objects between threads.

type BoxedDirectLDLSolver<T> = Box<dyn DirectLDLSolver<T> + Send>;

pub struct DirectLDLKKTSolver<T> {
    // problem dimensions
    m: usize,
    n: usize,
    p: usize,

    // Left and right hand sides for solves
    x: Vec<T>,
    b: Vec<T>,

    // internal workspace for IR scheme
    // and static offsetting of KKT
    work1: Vec<T>,
    work2: Vec<T>,

    // KKT mapping from problem data to KKT
    map: LDLDataMap,

    // the expected signs of D in KKT = LDL^T
    dsigns: Vec<i8>,

    // a vector for storing the entries of Hs blocks
    // on the KKT matrix block diagonal
    Hsblocks: Vec<T>,

    //unpermuted KKT matrix
    KKT: CscMatrix<T>,

    // the direct linear LDL solver
    ldlsolver: BoxedDirectLDLSolver<T>,

    // the diagonal regularizer currently applied
    diagonal_regularizer: T,
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    pub fn new(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        m: usize,
        n: usize,
        settings: &CoreSettings<T>,
    ) -> Self {
        // solving in sparse format.  Need this many
        // extra variables for SOCs
        let p = 2 * cones.type_count(SupportedConeTag::SecondOrderCone);

        // LHS/RHS/work for iterative refinement
        let x = vec![T::zero(); n + m + p];
        let b = vec![T::zero(); n + m + p];
        let work1 = vec![T::zero(); n + m + p];
        let work2 = vec![T::zero(); n + m + p];

        // the expected signs of D in LDL
        let mut dsigns = vec![1_i8; n + m + p];
        _fill_signs(&mut dsigns, m, n, p);

        // updates to the diagonal of KKT will be
        // assigned here before updating matrix entries
        let Hsblocks = allocate_kkt_Hsblocks::<T, T>(cones);

        // get a constructor for the LDL solver we should use,
        // and also the matrix shape it requires
        let (kktshape, ldl_ctor) = _get_ldlsolver_config(settings);

        //construct a KKT matrix of the right shape
        let (KKT, map) = assemble_kkt_matrix(P, A, cones, kktshape);

        let diagonal_regularizer = T::zero();

        // now make the LDL linear solver engine
        let ldlsolver = ldl_ctor(&KKT, &dsigns, settings);

        Self {
            m,
            n,
            p,
            x,
            b,
            work1,
            work2,
            map,
            dsigns,
            Hsblocks,
            KKT,
            ldlsolver,
            diagonal_regularizer,
        }
    }
}

impl<T> KKTSolver<T> for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool {
        let map = &self.map;

        // Set the elements the W^tW blocks in the KKT matrix.
        cones.get_Hs(&mut self.Hsblocks);

        let (values, index) = (&mut self.Hsblocks, &map.Hsblocks);
        // change signs to get -W^TW
        values.negate();
        _update_values(&mut self.ldlsolver, &mut self.KKT, index, values);

        // update the scaled u and v columns.
        let mut cidx = 0; // which of the SOCs are we working on?

        for cone in cones.iter() {
            // `cone` here will be of our SupportedCone enum wrapper, so
            //  we can extract a SecondOrderCone `soc`
            if let SupportedCone::SecondOrderCone(soc) = cone {
                let η2 = T::powi(soc.η, 2);

                //off diagonal columns (or rows)s
                let KKT = &mut self.KKT;
                let ldlsolver = &mut self.ldlsolver;

                _update_values(ldlsolver, KKT, &map.SOC_u[cidx], &soc.u);
                _update_values(ldlsolver, KKT, &map.SOC_v[cidx], &soc.v);
                _scale_values(ldlsolver, KKT, &map.SOC_u[cidx], -η2);
                _scale_values(ldlsolver, KKT, &map.SOC_v[cidx], -η2);

                //add η^2*(-1/1) to diagonal in the extended rows/cols
                _update_values(ldlsolver, KKT, &[map.SOC_D[cidx * 2]], &[-η2; 1]);
                _update_values(ldlsolver, KKT, &[map.SOC_D[cidx * 2 + 1]], &[η2; 1]);

                cidx += 1;
            } //end match
        } //end for

        self.regularize_and_refactor(settings)
    } //end fn

    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]) {
        let (m, n, p) = (self.m, self.n, self.p);

        self.b[0..n].copy_from(rhsx);
        self.b[n..(n + m)].copy_from(rhsz);
        self.b[n + m..(n + m + p)].fill(T::zero());
    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        self.ldlsolver.solve(&mut self.x, &self.b);

        let is_success = {
            if settings.iterative_refinement_enable {
                self.iterative_refinement(settings)
            } else {
                self.x.is_finite()
            }
        };

        if is_success {
            self.getlhs(lhsx, lhsz);
        }

        is_success
    }
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    // extra helper functions, not required for KKTSolver trait
    fn getlhs(&self, lhsx: Option<&mut [T]>, lhsz: Option<&mut [T]>) {
        let x = &self.x;
        let (m, n, _p) = (self.m, self.n, self.p);

        if let Some(v) = lhsx {
            v.copy_from(&x[0..n]);
        }
        if let Some(v) = lhsz {
            v.copy_from(&x[n..(n + m)]);
        }
    }

    fn regularize_and_refactor(&mut self, settings: &CoreSettings<T>) -> bool {
        let map = &self.map;
        let KKT = &mut self.KKT;
        let dsigns = &self.dsigns;
        let diag_kkt = &mut self.work1;
        let diag_shifted = &mut self.work2;

        if settings.static_regularization_enable {
            // hold a copy of the true KKT diagonal
            // diag_kkt .= KKT.nzval[map.diag_full];
            for (d, idx) in zip(&mut *diag_kkt, &map.diag_full) {
                *d = KKT.nzval[*idx];
            }

            let eps = _compute_regularizer(diag_kkt, settings);

            // compute an offset version, accounting for signs
            diag_shifted.copy_from(diag_kkt);

            zip(&mut *diag_shifted, dsigns).for_each(|(shift, &sign)| {
                if sign == 1 {
                    *shift += eps;
                } else {
                    *shift -= eps;
                }
            });

            // overwrite the diagonal of KKT and within the ldlsolver
            _update_values(&mut self.ldlsolver, KKT, &map.diag_full, diag_shifted);

            // remember the value we used.  Not needed,
            // but possibly useful for debugging
            self.diagonal_regularizer = eps;
        }

        //refactor with new data
        let is_success = self.ldlsolver.refactor(KKT);

        if settings.static_regularization_enable {
            // put our internal copy of the KKT matrix back the way
            // it was. Not necessary to fix the ldlsolver copy because
            // this is only needed for our post-factorization IR scheme

            _update_values_KKT(KKT, &map.diag_full, diag_kkt);
        }

        is_success
    }

    fn iterative_refinement(&mut self, settings: &CoreSettings<T>) -> bool {
        let (x, b) = (&mut self.x, &self.b);
        let (e, dx) = (&mut self.work1, &mut self.work2);

        // iterative refinement params
        let reltol = settings.iterative_refinement_reltol;
        let abstol = settings.iterative_refinement_abstol;
        let maxiter = settings.iterative_refinement_max_iter;
        let stopratio = settings.iterative_refinement_stop_ratio;

        let K = &self.KKT;
        let normb = b.norm_inf();

        //compute the initial error
        let mut norme = _get_refine_error(e, b, K, x);

        for _ in 0..maxiter {
            // bail on numerical error
            if !norme.is_finite() {
                return false;
            }

            if norme <= (abstol + reltol * normb) {
                //within tolerance.  Exit
                break;
            }

            let lastnorme = norme;

            //make a refinement
            self.ldlsolver.solve(dx, e);

            //prospective solution is x + dx.  Use dx space to
            // hold it for a check before applying to x
            dx.axpby(T::one(), x, T::one()); //now dx is really x + dx
            norme = _get_refine_error(e, b, K, dx);

            let improved_ratio = lastnorme / norme;
            if improved_ratio < stopratio {
                //insufficient improvement.  Exit
                if improved_ratio > T::one() {
                    //swap instead of copying to x
                    std::mem::swap(x, dx);
                }
                break;
            } else {
                //swap instead of copying to x
                std::mem::swap(x, dx);
            }
        }
        //NB: "success" means only that we had a finite valued result
        true
    }
}

fn _compute_regularizer<T: FloatT>(diag_kkt: &[T], settings: &CoreSettings<T>) -> T {
    let maxdiag = diag_kkt.norm_inf();

    // Compute a new regularizer
    settings.static_regularization_constant + settings.static_regularization_proportional * maxdiag
}

//  computes e = b - Kξ, overwriting the first argument
//  and returning its norm

fn _get_refine_error<T: FloatT>(e: &mut [T], b: &[T], K: &CscMatrix<T>, ξ: &mut [T]) -> T {
    // Note that K is only triu data, so need to
    // be careful when computing the residual here

    e.copy_from(b);
    K.symv(e, ξ, -T::one(), T::one()); //#  e = b - Kξ

    e.norm_inf()
}

type LDLConstructor<T> = fn(&CscMatrix<T>, &[i8], &CoreSettings<T>) -> BoxedDirectLDLSolver<T>;

fn _get_ldlsolver_config<T>(settings: &CoreSettings<T>) -> (MatrixTriangle, LDLConstructor<T>)
where
    T: FloatT,
{
    //The Julia version implements this using a module scope dictionary,
    //which allows users to register custom solver types.  That seems much
    //harder to do in Rust since a static mutable Hashmap is unsafe.  For
    //now, we use a fixed lookup table, so any new suppored solver types
    //supported must be added here.   It should be possible to allow a
    //"custom" LDL solver in the settings in well, whose constructor and
    //and matrix shape could then be registered as some (probably hidden)
    //options in the settings.

    let ldlptr: LDLConstructor<T>;
    let kktshape: MatrixTriangle;

    match settings.direct_solve_method.as_str() {
        "qdldl" => {
            kktshape = QDLDLDirectLDLSolver::<T>::required_matrix_shape();
            ldlptr = |M, D, S| Box::new(QDLDLDirectLDLSolver::<T>::new(M, D, S));
        }
        "custom" => {
            unimplemented!();
        }
        _ => {
            panic! {"Unrecognized LDL solver type"};
        }
    }
    (kktshape, ldlptr)
}

// update entries of the KKT matrix using the given index into its CSC representation.
// applied to both the unpermuted matrix of the kktsolver and also to the ldlsolver
fn _update_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    values: &[T],
) {
    //Update values in the KKT matrix K
    _update_values_KKT(KKT, index, values);

    // give the LDL subsolver an opportunity to update the same
    // values if needed.   This latter is useful for QDLDL since
    // it stores its own permuted copy internally
    ldlsolver.update_values(index, values);
}

fn _update_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], values: &[T]) {
    for (idx, v) in zip(index, values) {
        KKT.nzval[*idx] = *v;
    }
}

fn _scale_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    scale: T,
) {
    //Update values in the KKT matrix K
    _scale_values_KKT(KKT, index, scale);

    // ...and in the LDL subsolver if needed
    ldlsolver.scale_values(index, scale);
}

//scales KKT matrix values
fn _scale_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], scale: T) {
    for idx in index.iter() {
        KKT.nzval[*idx] *= scale;
    }
}

fn _fill_signs(signs: &mut [i8], m: usize, n: usize, p: usize) {
    signs.fill(1);

    //flip expected negative signs of D in LDL
    signs[n..(n + m)].iter_mut().for_each(|x| *x = -*x);

    //the trailing block of p entries should
    //have alternating signs
    signs[(n + m)..(n + m + p)]
        .iter_mut()
        .step_by(2)
        .for_each(|x| *x = -*x);
}
