#![allow(non_snake_case)]

use super::ldlsolvers::qdldl::*;
use super::*;
use crate::solver::core::kktsolvers::KKTSolver;
use crate::solver::core::{cones::*, CoreSettings};

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
    work_e: Vec<T>,
    work_dx: Vec<T>,

    // KKT mapping from problem data to KKT
    map: LDLDataMap,

    // the expected signs of D in KKT = LDL^T
    dsigns: Vec<i8>,

    // a vector for storing the entries of WtW blocks
    // on the KKT matrix block diagonal
    WtWblocks: Vec<T>,

    //unpermuted KKT matrix
    KKT: CscMatrix<T>,

    // the direct linear LDL solver
    ldlsolver: BoxedDirectLDLSolver<T>,
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
        let p = 2 * cones.type_count("SecondOrderConeT");

        // LHS/RHS/work for iterative refinement
        let x = vec![T::zero(); n + m + p];
        let b = vec![T::zero(); n + m + p];
        let work_e = vec![T::zero(); n + m + p];
        let work_dx = vec![T::zero(); n + m + p];

        // the expected signs of D in LDL
        let mut dsigns = vec![1_i8; n + m + p];
        _fill_signs(&mut dsigns, m, n, p);

        // updates to the diagonal of KKT will be
        // assigned here before updating matrix entries
        let WtWblocks = allocate_kkt_WtW_blocks::<T, T>(cones);

        // get a constructor for the LDL solver we should use,
        // and also the matrix shape it requires
        let (kktshape, ldl_ctor) = _get_ldlsolver_config(settings);

        //construct a KKT matrix of the right shape
        let (mut KKT, map) = assemble_kkt_matrix(P, A, cones, kktshape);

        if settings.static_regularization_enable {
            let eps = settings.static_regularization_eps;
            _offset_values_KKT(&mut KKT, &map.diag_full[0..n], eps, &dsigns[0..n]);
        }

        // now make the LDL linear solver engine
        let ldlsolver = ldl_ctor(&KKT, &dsigns, settings);

        Self {
            m,
            n,
            p,
            x,
            b,
            work_e,
            work_dx,
            map,
            dsigns,
            WtWblocks,
            KKT,
            ldlsolver,
        }
    }

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
}

type LDLConstructor<T> = fn(&CscMatrix<T>, &[i8], &CoreSettings<T>) -> BoxedDirectLDLSolver<T>;

fn _get_ldlsolver_config<T: FloatT>(
    settings: &CoreSettings<T>,
) -> (MatrixTriangle, LDLConstructor<T>)
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
    for (idx, v) in index.iter().zip(values.iter()) {
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

// offset diagonal entries of the KKT matrix over the Range
// of inices passed.  Length of signs and index must agree
fn _offset_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    offset: T,
    signs: &[i8],
) {
    assert_eq!(index.len(), signs.len());

    //Update values in the KKT matrix K.
    _offset_values_KKT(KKT, index, offset, signs);

    // ...and in the LDL subsolver if needed
    ldlsolver.offset_values(index, offset, signs);
}

fn _offset_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], offset: T, signs: &[i8]) {
    assert_eq!(index.len(), signs.len());

    for (&idx, &sign) in index.iter().zip(signs.iter()) {
        let sign = T::from_i8(sign).unwrap();
        KKT.nzval[idx] += sign * offset;
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

impl<T> KKTSolver<T> for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) {
        let map = &self.map;

        // Set the elements the W^tW blocks in the KKT matrix.
        cones.get_WtW_block(&mut self.WtWblocks);

        let (values, index) = (&mut self.WtWblocks, &map.WtWblocks);
        // change signs to get -W^TW
        values.negate();
        _update_values(&mut self.ldlsolver, &mut self.KKT, index, values);

        // update the scaled u and v columns.
        let mut cidx = 0; // which of the SOCs are we working on?

        for (i, cone) in cones.iter().enumerate() {
            if matches!(cones.types[i], SupportedCones::SecondOrderConeT(_)) {
                //here we need to recover the inner SOC value for
                //this cone so we can access its fields

                let K = cone.as_any().downcast_ref::<SecondOrderCone<T>>();

                match K {
                    None => {
                        panic!("cone type list is corrupt.");
                    }
                    Some(K) => {
                        let η2 = T::powi(K.η, 2);

                        //off diagonal columns (or rows)s
                        let KKT = &mut self.KKT;
                        let ldlsolver = &mut self.ldlsolver;

                        _update_values(ldlsolver, KKT, &map.SOC_u[cidx], &K.u);
                        _update_values(ldlsolver, KKT, &map.SOC_v[cidx], &K.v);
                        _scale_values(ldlsolver, KKT, &map.SOC_u[cidx], -η2);
                        _scale_values(ldlsolver, KKT, &map.SOC_v[cidx], -η2);

                        //add η^2*(-1/1) to diagonal in the extended rows/cols
                        _update_values(ldlsolver, KKT, &[map.SOC_D[cidx * 2]], &[-η2; 1]);
                        _update_values(ldlsolver, KKT, &[map.SOC_D[cidx * 2 + 1]], &[η2; 1]);

                        cidx += 1;
                    }
                } //end match
            } //end if SOC
        } //end for

        // Perturb the diagonal terms WtW that we have just overwritten
        // with static regularizers.  Note that we don't want to shift
        // elements in the ULHS (corresponding to P) since we already
        // shifted them at initialization and haven't overwritten them
        if settings.static_regularization_enable {
            let eps = settings.static_regularization_eps;
            let (m, n, p) = (self.m, self.n, self.p);
            _offset_values(
                &mut self.ldlsolver,
                &mut self.KKT,
                &map.diag_full[n..(n + m + p)],
                eps,
                &self.dsigns[n..(n + m + p)],
            );
        }

        //refactor with new data
        self.ldlsolver.refactor(&self.KKT);
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
    ) {
        self.ldlsolver.solve(&mut self.x, &self.b);

        if settings.iterative_refinement_enable {
            self.iterative_refinement(settings);
        }
        self.getlhs(lhsx, lhsz);
    }
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn iterative_refinement(&mut self, settings: &CoreSettings<T>) {
        let (x, b) = (&mut self.x, &self.b);
        let (e, dx) = (&mut self.work_e, &mut self.work_dx);

        // iterative refinement params
        let reltol = settings.iterative_refinement_reltol;
        let abstol = settings.iterative_refinement_abstol;
        let maxiter = settings.iterative_refinement_max_iter;
        let stopratio = settings.iterative_refinement_stop_ratio;

        let eps = {
            if settings.static_regularization_enable {
                settings.static_regularization_eps
            } else {
                T::zero()
            }
        };

        // Note that K is only triu data, so need to
        // be careful when computing the residual here
        let K = &self.KKT;
        let normb = b.norm_inf();

        //compute the initial error
        let mut norme = _get_refine_error(e, b, K, &self.dsigns, eps, x);

        for _ in 0..maxiter {
            if norme <= (abstol + reltol * normb) {
                //within tolerance.  Exit
                return;
            }

            let lastnorme = norme;

            //make a refinement
            self.ldlsolver.solve(dx, e);

            //prospective solution is x + dx.  Use dx space to
            // hold it for a check before applying to x
            dx.axpby(T::one(), x, T::one()); //now dx is really x + dx
            norme = _get_refine_error(e, b, K, &self.dsigns, eps, dx);

            if lastnorme / norme < stopratio {
                //insufficient improvement.  Exit
                return;
            } else {
                //just swap instead of copying to x
                std::mem::swap(x, dx);
            }
        }
    }
}

fn _get_refine_error<T: FloatT>(
    e: &mut [T],
    b: &[T],
    K: &CscMatrix<T>,
    dsigns: &[i8],
    eps: T,
    ξ: &mut [T],
) -> T {
    e.copy_from(b);
    K.symv(e, MatrixTriangle::Triu, ξ, -T::one(), T::one()); //#  e = b - Kξ

    if !T::is_zero(&eps) {
        //@. e += ϵ * D * ξ
        for (i, eval) in e.iter_mut().enumerate() {
            let s = T::from_i8(dsigns[i]).unwrap();
            *eval += eps * s * ξ[i];
        }
    }

    e.norm_inf()
}
