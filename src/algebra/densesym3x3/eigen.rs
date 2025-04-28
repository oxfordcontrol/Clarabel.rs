#![allow(non_snake_case)]
use crate::algebra::dense::DenseStorageMatrix;
use crate::algebra::*;

// 3x3 symmetric eigendecomposition using a hybrid analytic / Jacobi method.  See:

// Kopp, Joachim. "Efficient numerical diagonalization of hermitian 3× 3 matrices."
// International Journal of Modern Physics C 19.03 (2008): 523-548.

// Numerical recipes in C, 2nd edition, §11.1

type Matrix3x3<T> = DenseStorageMatrix<[T; 9], T>;
impl<T> Matrix3x3<T>
where
    T: FloatT,
{
    pub fn new() -> Self {
        Self {
            size: (3, 3),
            data: [T::zero(); 9],
            phantom: std::marker::PhantomData,
        }
    }
    pub fn zeros() -> Self {
        Self::new()
    }
}

impl<T> DenseMatrixSym3<T>
where
    T: FloatT,
{
    pub(crate) fn eigvals(&mut self) -> [T; 3] {
        // eigenvalues only, not target for V
        eigen_hybrid(self, &mut None)
    }

    pub(crate) fn eigen(&mut self, V: &mut Matrix3x3<T>) -> [T; 3] {
        // compute eigenvalues and eigenvectors
        eigen_hybrid(self, &mut Some(V))
    }
}

fn eigen_hybrid<T: FloatT>(
    A: &mut DenseMatrixSym3<T>,
    V: &mut Option<&mut Matrix3x3<T>>,
) -> [T; 3] {
    //first try analytic eigenvalues
    let w = eigvals_analytic(A);

    // check analytic vectors (and populate V if provided)
    if eigvecs_analytic(A, w, V).is_ok() {
        // analytic method worked, return the eigenvalues
        return w;
    } else {
        // analytic method failed, use Jacobi method
        eigen_jacobi(A, V)
    }
}

fn eigvals_analytic<T: FloatT>(A: &DenseMatrixSym3<T>) -> [T; 3] {
    // 3x3 analytic eigenvalue method
    // this is the method of Cardano

    // a bunch of constants for analytic method
    let C_27_OVER_2: T = (27.0 / 2.0).as_T();
    let C_27_OVER_4: T = (27.0 / 4.0).as_T();
    let ONE_THIRD: T = (1.0 / 3.0).as_T();
    let SQRT3_INV: T = T::recip(T::sqrt((3.0).as_T()));

    // 1. Compute the characteristic polynomial
    // 2. Find the roots of the polynomial
    // 3. Sort the roots in ascending order

    // Matrix is represented as :
    // A = [a b c]
    //     [* d e]
    //     [* * f]

    let A11 = A.data[0]; // == a == A[(0, 0)];
    let A12 = A.data[1]; // == b == A[(0, 1)];
    let A22 = A.data[2]; // == c == A[(1, 1)];
    let A13 = A.data[3]; // == d == A[(0, 2)];
    let A23 = A.data[4]; // == e == A[(1, 2)];
    let A33 = A.data[5]; // == f == A[(2, 2)];

    // precompute products
    let de = A12 * A23;
    let dd = A12 * A12;
    let ee = A23 * A23;
    let ff = A13 * A13;

    // Coefficients of the characteristic polynomial
    let m = A11 + A22 + A33;
    let c1 = (A11 * (A22 + A33) + A22 * A33) - (dd + ee + ff);
    let c0 = A33 * dd + A11 * ee + A22 * ff - A11 * A22 * A33 - A13 * de * (2.0).as_T();

    // p, q and sqrt(p)
    let p = m * m - c1 * (3.0).as_T();
    let q = m * (p - c1 * (1.5).as_T()) - C_27_OVER_2 * c0;
    let sqrt_p = T::sqrt(T::abs(p));

    // phi
    let phi = ((c1 * c1) * (p - c1) * (0.25).as_T() + c0 * (q + C_27_OVER_4 * c0)) * (27.0).as_T();
    let phi = ONE_THIRD * T::atan2(T::sqrt(T::abs(phi)), q);

    // c and s
    let s = SQRT3_INV * sqrt_p * T::sin(phi);
    let c = sqrt_p * T::cos(phi);

    // eigenvalues (sorted small to large)
    let w1 = ONE_THIRD * (m - c);
    let w2 = w1 + s;
    let w0 = w1 + c;
    let w1 = w1 - s;

    [w1, w2, w0]
}

fn eigvecs_analytic<T: FloatT>(
    A: &DenseMatrixSym3<T>,
    w: [T; 3],
    V: &mut Option<&mut Matrix3x3<T>>,
) -> Result<(), ()> {
    // analytical computes eigenvectors given eigenvalues,
    // and returns false if the accuracy is not good enough.

    // NB: this function will only compute two vectors internally,
    // since it is only being used as a check for accuracy.  The
    // third vector can be computed as the cross product of the
    // first two if the vectors are actually required.

    // compute the threshold from Kopp eq 44
    let wmax = maxabs3(w);
    let wmaxsq = wmax * wmax;
    let tol = if wmax < T::one() {
        T::epsilon() * (256.).as_T() * wmaxsq
    } else {
        T::epsilon() * (256.).as_T() * (wmaxsq * wmaxsq)
    };

    // compute the first eigenvector (A1 - w1e1) x (A2 - w1e2)
    let a = [
        A.data[0] - w[0], // A11 - w1
        A.data[1],        // A21 == A12
        A.data[3],        // A31 == A13
    ];
    let b = [
        A.data[1],        // A12
        A.data[2] - w[0], // A22 - w1
        A.data[4],        // A32 == A23
    ];
    let mut v1 = cross3(a, b);
    let normv1 = norm3(v1);

    if normv1 < tol {
        return Err(()); //fail
    }

    // compute the second eigenvector (A1 - w1e1) x (A2 - w1e2)
    let a = [
        A.data[0] - w[1], // A11 - w1
        A.data[1],        // A21 == A12
        A.data[3],        // A31 == A13
    ];
    let b = [
        A.data[1],        // A12
        A.data[2] - w[1], // A22 - w1
        A.data[4],        // A32 == A23
    ];
    let mut v2 = cross3(a, b);
    let normv2 = norm3(v2);
    if normv2 < tol {
        return Err(()); //fail
    }

    if let Some(vecs) = V {
        // normalize the vectors and store them in V
        scale3(&mut v1, normv1.recip());
        scale3(&mut v2, normv2.recip());
        let v3 = cross3(v1, v2);

        vecs.data[0..3].copy_from_slice(&v1);
        vecs.data[3..6].copy_from_slice(&v2);
        vecs.data[6..9].copy_from_slice(&v3);
    }

    Ok(()) //success
}

fn eigen_jacobi<T: FloatT>(
    A: &mut DenseMatrixSym3<T>,
    V: &mut Option<&mut Matrix3x3<T>>,
) -> [T; 3] {
    // Jacobi method for eigenvalue computation

    if let Some(vecs) = V {
        // set to identity
        V.data.set(T::zero());
        V.data[0] = T::one();
        V.data[4] = T::one();
        V.data[8] = T::one();
    }

    let tol = T::epsilon() * (16.).as_T();

    // benchmarks never exceed 12 iterations in f64
    for _ in 1..=100 {
        let (idx, max_off_diag) = find_largest_off_diag(&A);

        // check convergence
        if max_off_diag < tol {
            break;
        }

        // compute the Jacobi rotation
        let (c, s, t) = compute_rotation(&A, idx);

        // apply rotation to A (and update eigenvectors...)
        apply_rotation(A, idx, c, s, t);

        if let Some(vecs) = V {
            // update the eigenvectors
            update_eigenvectors(vecs, idx, c, s);
        }
    }
    //eigenvalues are on the diagonal of A
    // return sorted as a vector
    let w = {
        if let Some(vecs) = V {
            sort3_with_columns(A.data[0], A.data[2], A.data[5], vecs)
        } else {
            sort3(A.data[0], A.data[2], A.data[5])
        }
    };
    w
}

fn find_largest_off_diag<T: FloatT>(A: &DenseMatrixSym3<T>) -> (usize, T) {
    // here the idx will be index into the 6 element upper
    // triangle data of the 3x3 matrix

    let abs12 = A.data[1].abs();
    let abs13 = A.data[3].abs();
    let abs23 = A.data[4].abs();

    let mut idx = 1;
    let mut max_off_diag = abs12;

    if abs13 > max_off_diag {
        (idx, max_off_diag) = (3, abs13);
    }

    if abs23 > max_off_diag {
        (idx, max_off_diag) = (4, abs23);
    }

    (idx, max_off_diag)
}

fn compute_rotation<T: FloatT>(A: &DenseMatrixSym3<T>, idx: usize) -> (T, T, T) {
    // compute the Jacobi rotation for the given index

    // the idx is a linear index into the upper triangle.   For
    // each element on the triangle, need to work on the corresponding
    // diagonal terms
    let (Apq, App, Aqq) = match idx {
        1 => (A.data[1], A.data[0], A.data[2]), // A12, A11, A22
        3 => (A.data[3], A.data[0], A.data[5]), // A[0, 2], A[0, 0], A[2, 2],
        4 => (A.data[4], A.data[2], A.data[5]), // A[1, 1], A[2, 2], A[1, 2]
        _ => unreachable!("Invalid index for Jacobi rotation"),
    };

    if Apq == T::zero() {
        return (T::one(), T::zero(), T::zero());
    }

    let diffdiag = Aqq - App;
    let absApp = App.abs();
    let absAqq = Aqq.abs();

    let d = if absApp > absAqq { absApp } else { absAqq }; //a bit faster than f64::max, assumes finite
    if d < T::epsilon() || diffdiag.abs() < T::epsilon() * d {
        // diagonal elements nearly equal, use 45 degrees
        let c = T::FRAC_1_SQRT_2();
        let s = T::FRAC_1_SQRT_2();
        return (c, s, T::one());
    } else {
        let theta = diffdiag / (Apq * (2.0).as_T());

        // need caution here if theta^2 overflows
        let thetasq = theta * theta;
        let t = {
            if thetasq.is_finite() {
                theta.signum() / (theta.abs() + T::sqrt(T::one() + thetasq))
            } else {
                let half: T = (0.5).as_T();
                half / theta
            }
        };
        let c = T::recip(T::sqrt(T::one() + t * t));
        let s = t * c;

        (c, s, t)
    }
}

fn apply_rotation<T: FloatT>(A: &mut DenseMatrixSym3<T>, idx: usize, c: T, s: T, t: T) {
    // apply the Jacobi rotation to the matrix A

    let tau = s / (c + T::one());
    let tApq = t * A.data[idx];

    match idx {
        1 => {
            // rotating away the A12 term
            let (A13, A23) = (A.data[3], A.data[4]);
            A.data[3] = A13 - s * (A23 + tau * A13);
            A.data[4] = A23 + s * (A13 - tau * A23);
            A.data[0] -= tApq;
            A.data[2] += tApq;
        }
        3 => {
            // the A13 term
            let (A12, A23) = (A.data[1], A.data[4]);
            A.data[1] = A12 - s * (A23 + tau * A12);
            A.data[4] = A23 + s * (A12 - tau * A23);
            A.data[0] -= tApq;
            A.data[5] += tApq;
        }
        4 => {
            // the A23 term
            let (A12, A13) = (A.data[1], A.data[3]);
            A.data[1] = A12 - s * (A13 + tau * A12);
            A.data[3] = A13 + s * (A12 - tau * A13);
            A.data[2] -= tApq;
            A.data[5] += tApq;
        }
        _ => unreachable!("Invalid index for Jacobi rotation"),
    }

    // eliminate the rotated off-diagonal
    A.data[idx] = T::zero();
}

fn update_eigenvectors<T: FloatT>(V: &mut Matrix3x3<T>, idx: usize, c: T, s: T) {
    // update the eigenvectors in V
    let (p, q) = match idx {
        1 => (0, 1),
        3 => (0, 2),
        4 => (1, 2),
        _ => unreachable!("Invalid index for Jacobi rotation"),
    };

    for i in 0..3 {
        let Vip = V[(i, p)];
        let Viq = V[(i, q)];
        V[(i, p)] = c * Vip - s * Viq;
        V[(i, q)] = s * Vip + c * Viq;
    }
}

#[inline(always)]
fn sort3<T: FloatT>(mut a: T, mut b: T, mut c: T) -> [T; 3] {
    // sort three values in ascending order, quickly
    if a > b {
        core::mem::swap(&mut a, &mut b);
    }
    if a > c {
        core::mem::swap(&mut a, &mut c);
    }
    if b > c {
        core::mem::swap(&mut b, &mut c);
    }
    [a, b, c]
}

#[inline(always)]
fn sort3_with_columns<T: FloatT>(mut a: T, mut b: T, mut c: T, M: &mut Matrix3x3<T>) -> [T; 3] {
    // sort three values in ascending order, quickly
    // apply the same sorting to the columns of a matrix

    // possible improvement if https://github.com/rust-lang/rust/issues/88539
    // is accepted to support a swap_unchecked function
    if a > b {
        core::mem::swap(&mut a, &mut b);
        M.data.swap(0, 3);
        M.data.swap(1, 4);
        M.data.swap(2, 5);
    }
    if a > c {
        core::mem::swap(&mut a, &mut c);
        M.data.swap(0, 6);
        M.data.swap(1, 7);
        M.data.swap(2, 8);
    }
    if b > c {
        core::mem::swap(&mut b, &mut c);
        M.data.swap(3, 6);
        M.data.swap(4, 7);
        M.data.swap(5, 8);
    }
    [a, b, c]
}

#[inline(always)]
fn maxabs3<T: FloatT>(v: [T; 3]) -> T {
    let mut out = v[0].abs();
    let nxt = v[1].abs();
    if nxt > out {
        out = nxt;
    }
    let nxt = v[2].abs();
    if nxt > out {
        out = nxt;
    }
    out
}

#[inline(always)]
fn scale3<T: FloatT>(v: &mut [T; 3], s: T) {
    v[0] *= s;
    v[1] *= s;
    v[2] *= s;
}

#[inline(always)]
fn cross3<T: FloatT>(v1: [T; 3], v2: [T; 3]) -> [T; 3] {
    [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]
}

#[inline(always)]
fn norm3<T: FloatT>(v: [T; 3]) -> T {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

// ---- tests and benchmarks ----

#[cfg(test)]
mod test {

    use super::*;
    use crate::algebra::DenseMatrixSym3;
    use itertools::iproduct;
    use lapack::Select2F64;
    use std::ops::Range;

    fn sym3_test_iter(
        rng: Range<i32>,
        fcn: fn(f64) -> f64,
    ) -> impl Iterator<Item = DenseMatrixSym3<f64>> {
        iproduct!(
            rng.clone(),
            rng.clone(),
            rng.clone(),
            rng.clone(),
            rng.clone(),
            rng.clone()
        )
        .map(move |(a, b, c, d, e, f)| {
            let data = [
                fcn(a as f64), // Convert Item -> f64, then apply f
                fcn(b as f64),
                fcn(c as f64),
                fcn(d as f64),
                fcn(e as f64),
                fcn(f as f64),
            ];
            DenseMatrixSym3 { data }
        })
    }

    fn matrix_example_nice() -> (DenseMatrixSym3<f64>, [f64; 3]) {
        #[rustfmt::skip]
        // A = 
        // 4.0  2.0  2.0
        // 2.0  3.0  1.0
        // 2.0  1.0  -3.0
        let A = DenseMatrixSym3 {
            data: [4.0, 2.0, 3.0, 2.0, 1.0, -3.0],
        };

        let trueD = [-3.565507919110752, 1.47313296881958, 6.092374950291167];
        (A, trueD)
    }

    fn matrix_example_degen() -> (DenseMatrixSym3<f64>, [f64; 3]) {
        #[rustfmt::skip]
        let A = DenseMatrixSym3 {
            data: [-5.0,-5.0,-5.0,-5.0,-5.0,-5.0],
        };

        let trueD = [-15.0, 0.0, 0.0];
        (A, trueD)
    }

    fn matrix_example_hard() -> (DenseMatrixSym3<f64>, [f64; 3]) {
        #[rustfmt::skip]
        // A = 
        // 1e+5  1e+2  1e+2
        // 1e+2  1e+5  1e-5
        // 1e+2  1e-5  1e+5

        let A = DenseMatrixSym3 {
            data: [1e5,1e2,1e5,1e2,1e-5,1e5],
        };

        // computed via Julia BigFloat + GenericLinearAlgebra.jl
        let trueD = [99858.57864876269, 99999.99999, 100141.42136123731];
        (A, trueD)
    }

    fn check_eigvals(D1: [f64; 3], D2: [f64; 3]) -> bool {
        for (d1, d2) in D1.iter().zip(D2.iter()) {
            if (d1 - d2).abs() > 1e-10 {
                return false;
            }
        }
        true
    }

    fn check_eigvecs(A: &DenseMatrixSym3<f64>, V: &Matrix3x3<f64>, w: &[f64; 3]) -> bool {
        let mut out = [0.; 3];

        for i in 0..3 {
            let mut v = V.col_slice(i).to_vec();
            A.mul(&mut out.as_mut_slice(), &v);
            v.axpby(1.0, &out, -w[i]);
            if v.norm() > 1e-12 {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_sort3() {
        let t = [1.0, 2.0, 3.0];
        assert_eq!(sort3(1.0, 2.0, 3.0), t);
        assert_eq!(sort3(1.0, 3.0, 2.0), t);
        assert_eq!(sort3(2.0, 1.0, 3.0), t);
        assert_eq!(sort3(2.0, 3.0, 1.0), t);
        assert_eq!(sort3(3.0, 1.0, 2.0), t);
        assert_eq!(sort3(3.0, 2.0, 1.0), t);
    }

    #[test]
    fn test_eigen_nice() {
        let mut V = Matrix3x3::<f64>::new();

        // check the analytic eigenvalue function
        let (A, trueD) = matrix_example_nice();
        let D = eigvals_analytic(&A);
        assert!(check_eigvals(D, trueD));
        assert!(eigvecs_analytic(&A, D, &mut Some(&mut V)).is_ok());

        // check the jacobi eigenvalue function
        let (A, trueD) = matrix_example_nice();
        let D = eigen_jacobi(&mut A.clone(), &mut Some(&mut V.clone()));
        assert!(check_eigvals(D, trueD));
        assert!(check_eigvecs(&A, &V, &D));
    }

    #[test]
    fn test_eigen_degen() {
        let mut V = Matrix3x3::<f64>::new();

        // check the analytic eigenvalue function
        let (A, trueD) = matrix_example_degen();
        let D = eigvals_analytic(&A);
        assert!(check_eigvals(D, trueD));
        // should fail since the eigenvectors are not unique
        assert!(eigvecs_analytic(&A, D, &mut Some(&mut V)).is_err());

        // check the hybrid method
        let D = eigen_hybrid(&mut A.clone(), &mut Some(&mut V));
        assert!(check_eigvals(D, trueD));
        assert!(check_eigvecs(&A, &V, &D));

        // check the jacobi eigenvalue function
        let (A, trueD) = matrix_example_degen();
        let D = eigen_jacobi(&mut A.clone(), &mut Some(&mut V.clone()));
        assert!(check_eigvals(D, trueD));
        assert!(check_eigvecs(&A, &V, &D));
    }

    #[test]
    #[should_panic]
    fn test_eigvals_hard_analytic() {
        // check the analytic eigenvalue function
        let (A, trueD) = matrix_example_hard();
        let D = eigvals_analytic(&A);
        assert!(check_eigvals(D, trueD));
    }

    #[test]
    fn test_eigvals_hard_jacobi() {
        let mut V = Matrix3x3::<f64>::new();

        // check the jacobi eigenvalue function
        let (A, trueD) = matrix_example_hard();
        let D = eigen_jacobi(&mut A.clone(), &mut Some(&mut V));
        assert!(check_eigvals(D, trueD));
        assert!(check_eigvecs(&A, &V, &D));
    }

    #[test]
    fn test_jacobi_rotation() {
        for idx in [1, 3, 4] {
            let (mut A, _) = matrix_example_nice();

            // compute and apply a jacobi rotation and check
            // that the corresponding term has been zeroed
            let (c, s, t) = compute_rotation(&A, idx);
            apply_rotation(&mut A, idx, c, s, t);
            assert!(
                A.data[idx].abs() < 1e-14,
                "Jacobi rotation failed to zero term {}",
                idx
            );
        }
    }

    #[test]
    fn test_eigvals_grid() {
        let rng = -5..6;
        let iter = sym3_test_iter(rng, |x| x as f64);
        let mut eng = EigEngine::<f64>::new(3);

        for As in iter {
            let mut Af: Matrix<f64> = As.clone().into();

            // compute true eigenvalues from blas
            eng.eigen(&mut Af).unwrap();
            let mut trueD = [0.0; 3];
            trueD.copy_from_slice(&eng.λ);

            let D1 = eigvals_analytic(&As);
            let D2 = eigen_jacobi(&mut As.clone(), &mut None);

            if !check_eigvals(trueD, D1) {
                panic!("Eigenvalue error analytic: A = {:?}", As.data);
            }
            if !check_eigvals(trueD, D2) {
                panic!("Eigenvalue error jacobi: A = {:?}", As.data);
            }
        }
    }

    #[test]
    fn test_eigvals_exp_grid() {
        let rng = -5..6;
        let iter = sym3_test_iter(rng, |x| (10_f64).powf(x as f64));
        let mut eng = EigEngine::<f64>::new(3);

        for As in iter {
            let mut Af: Matrix<f64> = As.clone().into();

            // compute true eigenvalues from blas
            eng.eigen(&mut Af).unwrap();
            let mut trueD = [0.0; 3];
            trueD.copy_from_slice(&eng.λ);

            let D2 = eigen_jacobi(&mut As.clone(), &mut None);

            check_eigvals(trueD, D2);
        }
    }
}
