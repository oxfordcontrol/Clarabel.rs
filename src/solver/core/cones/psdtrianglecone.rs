use super::*;
use crate::{
    algebra::*,
    solver::{core::ScalingStrategy, CoreSettings},
};

// ------------------------------------
// Positive Semidefinite Cone (Scaled triangular form)
// ------------------------------------

pub struct PSDConeData<T> {
    cholS: CholeskyEngine<T>,
    cholZ: CholeskyEngine<T>,
    SVD: SVDEngine<T>,
    Eig: EigEngine<T>,
    λ: Vec<T>,
    Λisqrt: Vec<T>,
    R: Matrix<T>,
    Rinv: Matrix<T>,
    kronRR: Matrix<T>,
    B: Matrix<T>,
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
            cholS: CholeskyEngine::<T>::new(n),
            cholZ: CholeskyEngine::<T>::new(n),
            SVD: SVDEngine::<T>::new((n, n)),
            Eig: EigEngine::<T>::new(n),

            λ: vec![T::zero(); n],
            Λisqrt: vec![T::zero(); n],
            R: Matrix::zeros((n, n)),
            Rinv: Matrix::zeros((n, n)),
            kronRR: Matrix::zeros((n * n, n * n)),
            B: Matrix::zeros((Bm, n * n)),
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
        let e: &[T];

        if z.is_empty() {
            α = T::max_value();
            e = &[T::zero(); 0];
        } else {
            let Z = &mut self.data.workmat1;
            _svec_to_mat(Z, z);
            self.data.Eig.eigvals(Z).expect("Eigval error");
            e = &self.data.Eig.λ;
            α = e.minimum();
        }

        let β = e.iter().fold(T::zero(), |s, x| s + T::max(*x, T::zero())); //= sum(e[e.>0])
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
        _svec_to_mat(S, s);
        _svec_to_mat(Z, z);

        //compute Cholesky factors
        f.cholS.cholesky(S).expect("Cholesky error"); //PJG: could rethrow error here
        f.cholZ.cholesky(Z).expect("Cholesky error");
        let (L1, L2) = (&f.cholS.L, &f.cholZ.L);

        // SVD of L2'*L1,
        let tmp = &mut f.workmat1;
        tmp.mul(&L2.t(), L1, T::one(), T::zero());
        f.SVD.svd(tmp).expect("SVD error");

        // assemble λ (diagonal), R and Rinv.
        f.λ.copy_from(&f.SVD.s);
        f.Λisqrt.copy_from(&f.λ).sqrt().recip();

        //f.R = L1*(f.SVD.V)*f.Λisqrt
        f.R.mul(L1, &f.SVD.Vt.t(), T::one(), T::zero());
        f.R.rscale(&f.Λisqrt);

        //f.Rinv .= f.Λisqrt*(f.SVD.U)'*L2'
        f.Rinv.mul(&f.SVD.U.t(), &L2.t(), T::one(), T::zero());
        f.Rinv.lscale(&f.Λisqrt);

        // we should compute here the upper triangular part
        // of the matrix Q* (RR^T) ⨂ (RR^T) * P.  The operator
        // P is a matrix that transforms a packed triangle to
        // a vectorized full matrix.

        f.kronRR.kron(&f.R, &f.R);

        // B .= Q'*kRR, where Q' is the svec operator
        for i in 0..f.B.ncols() {
            let M = ReshapedMatrix::from_slice(f.kronRR.col_slice(i), f.R.nrows(), f.R.nrows());
            let b = f.B.col_slice_mut(i);
            _mat_to_svec(b, &M);
        }

        // compute Hs = triu(B*B')
        f.Hs.syrk(&f.B, T::one(), T::zero());

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
        _mul_Wx_inner(
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
        let αz = _step_length_psd_component(workΔ, engine, d, Λisqrt, αmax);

        // d = Δs̃ = W^{-T}Δs
        _mul_Wx_inner(
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
        let αs = _step_length_psd_component(workΔ, engine, d, Λisqrt, αmax);

        (αz, αs)
    }

    fn compute_barrier(&mut self, _z: &[T], _s: &[T], _dz: &[T], _ds: &[T], _α: T) -> T {
        // We should return this, but in a smarter way.
        // This is not yet implemented, but would only
        // be required for problems mixing PSD and
        // asymmetric cones
        //
        // return -log(det(s)) - log(det(z))
        unimplemented!("Mixed PSD and Exponential/Power cones are not yet supported");
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

        _svec_to_mat(X, x);
        _svec_to_mat(Z, z);

        let λ = &self.data.λ;
        let two: T = (2.).as_T();
        for i in 0..self.n {
            for j in 0..self.n {
                X[(i, j)] = (two * Z[(i, j)]) / (λ[i] + λ[j]);
            }
        }
        _mat_to_svec(x, X);
    }

    fn mul_W(&mut self, is_transpose: MatrixShape, y: &mut [T], x: &[T], α: T, β: T) {
        _mul_Wx_inner(
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
        _mul_Wx_inner(
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
fn _mul_Wx_inner<T>(
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
    _svec_to_mat(X, x);
    _svec_to_mat(Y, y);

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
    _mat_to_svec(y, Y);
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
        _svec_to_mat(Y, y);
        _svec_to_mat(Z, z);

        // X .= (Y*Z + Z*Y)/2
        // NB: works b/c Y and Z are both symmetric
        X.data_mut().set(T::zero()); //X.sym() will assert is_triu
        X.syr2k(Y, Z, (0.5).as_T(), T::zero());
        _mat_to_svec(x, &X.sym());
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

fn _step_length_psd_component<T>(
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
            _svec_to_mat(workΔ, d);
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

fn _svec_to_mat<T: FloatT>(M: &mut Matrix<T>, x: &[T]) {
    let mut idx = 0;
    for col in 0..M.ncols() {
        for row in 0..=col {
            if row == col {
                M[(row, col)] = x[idx];
            } else {
                M[(row, col)] = x[idx] * T::FRAC_1_SQRT_2();
                M[(col, row)] = x[idx] * T::FRAC_1_SQRT_2();
            }
            idx += 1;
        }
    }
}

//PJG : Perhaps implementation for Symmetric type would be faster
fn _mat_to_svec<MAT, T: FloatT>(x: &mut [T], M: &MAT)
where
    MAT: DenseMatrix<T = T, Output = T>,
{
    let mut idx = 0;
    for col in 0..M.ncols() {
        for row in 0..=col {
            x[idx] = {
                if row == col {
                    M[(row, col)]
                } else {
                    (M[(row, col)] + M[(col, row)]) * T::FRAC_1_SQRT_2()
                }
            };
            idx += 1;
        }
    }
}

#[test]

fn test_svec_conversions() {
    let n = 3;

    let X = Matrix::from(&[
        [1., 3., -2.], //
        [3., -4., 7.], //
        [-2., 7., 5.], //
    ]);

    let Y = Matrix::from(&[
        [2., 5., -4.],  //
        [5., 6., 2.],   //
        [-4., 2., -3.], //
    ]);

    let mut Z = Matrix::zeros((3, 3));

    let mut x = vec![0.; triangular_number(n)];
    let mut y = vec![0.; triangular_number(n)];

    // check inner product identity
    _mat_to_svec(&mut x, &X);
    _mat_to_svec(&mut y, &Y);

    assert!(f64::abs(x.dot(&y) - X.data().dot(Y.data())) < 1e-12);

    // check round trip
    _mat_to_svec(&mut x, &X);
    _svec_to_mat(&mut Z, &x);
    assert!(X.data().norm_inf_diff(Z.data()) < 1e-12);
}
