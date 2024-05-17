use super::*;
use crate::algebra::*;

// ------------------------------------
// Positive Semidefinite Cone (Scaled triangular form)
// ------------------------------------

pub struct PSDConeData<T> {
    chol1: CholeskyEngine<T>,
    chol2: CholeskyEngine<T>,
    SVD: SVDEngine<T>,
    Eig: EigEngine<T>,
    λ: Vec<T>,
    Λisqrt: Vec<T>,
    R: Matrix<T>,
    Rinv: Matrix<T>,
    Hs: Matrix<T>,

    //workspace for various internal uses
    workmat1: Matrix<T>,
    workmat2: Matrix<T>,
    workmat3: Matrix<T>,
    workvec: Vec<T>,
}

impl<T> PSDConeData<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        let Bm = triangular_number(n);

        Self {
            chol1: CholeskyEngine::<T>::new(n),
            chol2: CholeskyEngine::<T>::new(n),
            SVD: SVDEngine::<T>::new((n, n)),
            Eig: EigEngine::<T>::new(n),

            λ: vec![T::zero(); n],
            Λisqrt: vec![T::zero(); n],
            R: Matrix::zeros((n, n)),
            Rinv: Matrix::zeros((n, n)),
            Hs: Matrix::zeros((Bm, Bm)),

            //workspace for various internal uses
            workmat1: Matrix::zeros((n, n)),
            workmat2: Matrix::zeros((n, n)),
            workmat3: Matrix::zeros((n, n)),
            workvec: vec![T::zero(); triangular_number(n)],
        }
    }
}

pub struct PSDTriangleCone<T> {
    n: usize,                  // matrix dimension, i.e. matrix is n × n
    numel: usize,              // total number of elements in (lower triangle of) the matrix
    data: Box<PSDConeData<T>>, // Boxed so that the PSDCone enum_dispatch variant isn't huge
}

impl<T> PSDTriangleCone<T>
where
    T: FloatT,
{
    pub fn new(n: usize) -> Self {
        // n >= 0 guaranteed by type limit
        Self {
            n,
            numel: triangular_number(n),
            data: Box::new(PSDConeData::<T>::new(n)),
        }
    }
}

impl<T> Cone<T> for PSDTriangleCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        self.n
    }

    fn numel(&self) -> usize {
        self.numel
    }

    fn is_symmetric(&self) -> bool {
        true
    }

    fn is_sparse_expandable(&self) -> bool {
        false
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        true
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e).recip().scale(e.mean());
        true // scalar equilibration
    }

    // functions relating to unit vectors and cone initialization
    fn margins(&mut self, z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        let α: T;
        let β: T;

        if z.is_empty() {
            α = T::max_value();
            β = T::zero();
        } else {
            let Z = &mut self.data.workmat1;
            svec_to_mat(Z, z);
            self.data.Eig.eigvals(Z).expect("Eigval error");
            let e = &self.data.Eig.λ;
            α = e.minimum();
            β = e.iter().fold(T::zero(), |s, x| s + T::max(*x, T::zero())); //= sum(e[e.>0])
        }

        (α, β)
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, _pd: PrimalOrDualCone) {
        //adds αI to the vectorized triangle,
        //at elements [1,3,6....n(n+1)/2]
        for k in 0..self.n {
            z[triangular_index(k)] += α
        }
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        s.fill(T::zero());
        z.fill(T::zero());
        self.scaled_unit_shift(s, T::one(), PrimalOrDualCone::PrimalCone);
        self.scaled_unit_shift(z, T::one(), PrimalOrDualCone::DualCone);
    }

    fn set_identity_scaling(&mut self) {
        self.data.R.set_identity();
        self.data.Rinv.set_identity();
        self.data.Hs.set_identity();
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        _μ: T,
        _scaling_strategy: ScalingStrategy,
    ) -> bool {
        if s.is_empty() {
            //bail early on zero length cone
            return true;
        }

        let f = &mut self.data;
        let (S, Z) = (&mut f.workmat1, &mut f.workmat2);
        svec_to_mat(S, s);
        svec_to_mat(Z, z);

        //compute Cholesky factors
        let c1 = f.chol1.factor(S);
        let c2 = f.chol2.factor(Z);

        // bail if the cholesky factorization fails
        // PJG: Need proper Result return type here
        if c1.is_err() || c2.is_err() {
            return false;
        }

        let (L1, L2) = (&f.chol1.L, &f.chol2.L);

        // SVD of L2'*L1,
        let tmp = &mut f.workmat1;
        tmp.mul(&L2.t(), L1, T::one(), T::zero());
        f.SVD.factor(tmp).expect("SVD error");

        // assemble λ (diagonal), R and Rinv.
        f.λ.copy_from(&f.SVD.s);
        f.Λisqrt.copy_from(&f.λ).sqrt().recip();

        //f.R = L1*(f.SVD.V)*f.Λisqrt
        f.R.mul(L1, &f.SVD.Vt.t(), T::one(), T::zero());
        f.R.rscale(&f.Λisqrt);

        //f.Rinv .= f.Λisqrt*(f.SVD.U)'*L2'
        f.Rinv.mul(&f.SVD.U.t(), &L2.t(), T::one(), T::zero());
        f.Rinv.lscale(&f.Λisqrt);

        // compute R*R^T (upper triangular part only)
        let RRt = &mut f.workmat1;
        RRt.data_mut().set(T::zero());
        RRt.syrk(&f.R, T::one(), T::zero());

        // PJG: it is possibly faster to compute the whole of RRt, and not
        // just the upper triangle using syrk!, because then skron! can be
        // be called with a Matrix type instead of Symmetric.   The internal
        // indexing within skron! is then more straightforward and probably
        // faster.   Possibly also worth considering a version of skron!
        // that uses unchecked indexing.
        skron(&mut f.Hs, &RRt.sym());

        true //PJG: Should return result, with "?" operators above
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        self.data.Hs.sym().pack_triu(Hsblock);
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], work: &mut [T]) {
        // PJG: Why this way instead of Hs.sym() * x?
        self.mul_W(MatrixShape::N, work, x, T::one(), T::zero()); // work = Wx
        self.mul_W(MatrixShape::T, y, work, T::one(), T::zero()); // y = c Wᵀwork = W^TWx
    }

    fn affine_ds(&self, ds: &mut [T], _s: &[T]) {
        ds.set(T::zero());
        for k in 0..self.n {
            ds[triangular_index(k)] = self.data.λ[k] * self.data.λ[k];
        }
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        self._combined_ds_shift_symmetric(shift, step_z, step_s, σμ);
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], _z: &[T]) {
        self._Δs_from_Δz_offset_symmetric(out, ds, work);
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        _z: &[T],
        _s: &[T],
        _settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let Λisqrt = &self.data.Λisqrt;
        let d = &mut self.data.workvec;
        let engine = &mut self.data.Eig;

        // d = Δz̃ = WΔz
        mul_Wx_inner(
            MatrixShape::N,
            d,
            dz,
            T::one(),
            T::zero(),
            &self.data.R,
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        );
        let workΔ = &mut self.data.workmat1;
        let αz = step_length_psd_component(workΔ, engine, d, Λisqrt, αmax);

        // d = Δs̃ = W^{-T}Δs
        mul_Wx_inner(
            MatrixShape::T,
            d,
            ds,
            T::one(),
            T::zero(),
            &self.data.Rinv,
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        );
        let workΔ = &mut self.data.workmat1;
        let αs = step_length_psd_component(workΔ, engine, d, Λisqrt, αmax);

        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();
        barrier -= self.logdet_barrier(z, dz, α);
        barrier -= self.logdet_barrier(s, ds, α);
        barrier
    }
}

impl<T> PSDTriangleCone<T>
where
    T: FloatT,
{
    fn logdet_barrier(&mut self, x: &[T], dx: &[T], α: T) -> T
    where
        T: FloatT,
    {
        let (Q, q) = (&mut self.data.workmat1, &mut self.data.workvec);
        q.waxpby(T::one(), x, α, dx);
        svec_to_mat(Q, q);

        match self.data.chol1.factor(Q) {
            Ok(_) => self.data.chol1.logdet(),
            Err(_) => T::infinity(),
        }
    }
}

// ---------------------------------------------
// operations supported by symmetric cones only
// ---------------------------------------------

impl<T> SymmetricCone<T> for PSDTriangleCone<T>
where
    T: FloatT,
{
    // implements x = λ \ z for the SDP cone
    fn λ_inv_circ_op(&mut self, x: &mut [T], z: &[T]) {
        let X = &mut self.data.workmat1;
        let Z = &mut self.data.workmat2;

        svec_to_mat(X, x);
        svec_to_mat(Z, z);

        let λ = &self.data.λ;
        let two: T = (2.).as_T();
        for i in 0..self.n {
            for j in 0..self.n {
                X[(i, j)] = (two * Z[(i, j)]) / (λ[i] + λ[j]);
            }
        }
        mat_to_svec(x, X);
    }

    fn mul_W(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        mul_Wx_inner(
            is_transpose,
            y,
            x,
            α,
            β,
            &self.data.R,
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        )
    }

    fn mul_Winv(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        mul_Wx_inner(
            is_transpose,
            y,
            x,
            α,
            β,
            &self.data.Rinv,
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn mul_Wx_inner<T>(
    is_transpose: MatrixShape,
    y: &mut [T],
    x: &[T],
    α: T,
    β: T,
    Rx: &Matrix<T>,
    workmat1: &mut Matrix<T>,
    workmat2: &mut Matrix<T>,
    workmat3: &mut Matrix<T>,
) where
    T: FloatT,
{
    let (X, Y, tmp) = (workmat1, workmat2, workmat3);
    svec_to_mat(X, x);
    svec_to_mat(Y, y);

    match is_transpose {
        MatrixShape::T => {
            // Y .= α*(R*X*R') + βY        #W^T*x,   or....
            // Y .= α*(Rinv*X*Rinv') + βY  #W^{-T}*x
            tmp.mul(X, &Rx.t(), T::one(), T::zero());
            Y.mul(Rx, tmp, α, β);
        }
        MatrixShape::N => {
            // Y .= α*(R'*X*R) + βY         #W*x
            // Y .= α*(Rinv'*X*Rinv) + βY   #W^{-1}*x
            tmp.mul(&Rx.t(), X, T::one(), T::zero());
            Y.mul(tmp, Rx, α, β);
        }
    }
    mat_to_svec(y, Y);
}

// ---------------------------------------------
// Jordan algebra operations for symmetric cones
// ---------------------------------------------

impl<T> JordanAlgebra<T> for PSDTriangleCone<T>
where
    T: FloatT,
{
    fn circ_op(&mut self, x: &mut [T], y: &[T], z: &[T]) {
        let (Y, Z, X) = (
            &mut self.data.workmat1,
            &mut self.data.workmat2,
            &mut self.data.workmat3,
        );
        svec_to_mat(Y, y);
        svec_to_mat(Z, z);

        // X .= (Y*Z + Z*Y)/2
        // NB: works b/c Y and Z are both symmetric
        X.data_mut().set(T::zero()); //X.sym() will assert is_triu
        X.syr2k(Y, Z, (0.5).as_T(), T::zero());
        mat_to_svec(x, &X.sym());
    }

    fn inv_circ_op(&mut self, _x: &mut [T], _y: &[T], _z: &[T]) {
        // X should be the solution to (YX + XY)/2 = Z

        //  For general arguments this requires solution to a symmetric
        // Sylvester equation.  Throwing an error here since I do not think
        // the inverse of the ∘ operator is ever required for general arguments,
        // and solving this equation is best avoided.
        unreachable!();
    }
}

//-----------------------------------------
// internal operations for SDP cones
// ----------------------------------------

fn step_length_psd_component<T>(
    workΔ: &mut Matrix<T>,
    engine: &mut EigEngine<T>,
    d: &[T],
    Λisqrt: &[T],
    αmax: T,
) -> T
where
    T: FloatT,
{
    let γ = {
        if d.is_empty() {
            T::max_value()
        } else {
            svec_to_mat(workΔ, d);
            workΔ.lrscale(Λisqrt, Λisqrt);
            engine.eigvals(workΔ).expect("Eigval error");
            engine.λ.minimum()
        }
    };

    if γ < T::zero() {
        T::min(-γ.recip(), αmax)
    } else {
        αmax
    }
}

// produce the upper triangular part of the Symmetric Kronecker product of
// a symmtric matrix A with itself, i.e. triu(A ⊗_s A)
fn skron<T>(out: &mut Matrix<T>, A: &Symmetric<Matrix<T>>)
where
    T: FloatT,
{
    let sqrt2 = T::SQRT_2();
    let n = A.nrows();

    let mut col = 0;
    for l in 0..n {
        for k in 0..=l {
            let mut row = 0;
            let kl_eq = k == l;

            for j in 0..n {
                let Ajl = A[(j, l)];
                let Ajk = A[(j, k)];

                for i in 0..=j {
                    if row > col {
                        break;
                    }

                    let ij_eq = i == j;

                    out[(row, col)] = {
                        match (ij_eq, kl_eq) {
                            (false, false) => A[(i, k)] * Ajl + A[(i, l)] * Ajk,
                            (true, false) => sqrt2 * Ajl * Ajk,
                            (false, true) => sqrt2 * A[(i, l)] * Ajk,
                            (true, true) => Ajl * Ajl,
                        }
                    };

                    row += 1;
                } //end i
            } //end j
            col += 1;
        } //end k
    } //end l
}
