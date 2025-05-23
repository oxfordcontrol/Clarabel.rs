#![allow(non_snake_case)]
use crate::algebra::*;
use super::eigen::*;

// 3x3 SVD using a two-sided Jacobi method.

impl<T> DenseMatrix3<T>
where
    T: FloatT,
{
    pub(crate) fn svd(&mut self, U: &mut DenseMatrix3<T>, V: &mut DenseMatrix3<T>) -> [T; 3] {
        // two-sided Jacobi method for SVD

        let A = self;
        U.set_identity();
        V.set_identity();

        let tol = T::epsilon() * (256.).as_T();

        // benchmarks never exceed 10 iterations in f64
        for _ in 1..=25 {
            let (p, q, max_off_diag) = find_largest_off_diag_nonsym(A);

            // check convergence
            if max_off_diag < tol {
                break;
            }

            let (cl, sl, cr, sr, d1, d2) =
                compute_two_sided_rotation(A[(p, p)], A[(q, p)], A[(p, q)], A[(q, q)]);

            apply_two_sided_rotation(A, p, q, cl, sl, cr, sr, d1, d2);

            // Update U and V
            update_eigenvectors(U, p, q, cl, sl);
            update_eigenvectors(V, p, q, cr, sr);
        }

        fix_order_and_signs_3x3(A[(0, 0)], A[(1, 1)], A[(2, 2)], U, V)
    }
}

// find largest off-diagonal pair in an asymmetric matrix
// returns:
// - (p,q), the indices of the largest off-diagonal
// - max_off_diag, the abs sum of the largest off-diagonal pair

fn find_largest_off_diag_nonsym<T: FloatT>(A: &DenseMatrix3<T>) -> (usize, usize, T) {
    let absA12 = A[(0, 1)].abs() + A[(1, 0)].abs();
    let absA13 = A[(0, 2)].abs() + A[(2, 0)].abs();
    let absA23 = A[(1, 2)].abs() + A[(2, 1)].abs();
    let (mut p, mut q, mut max_off_diag) = (0, 1, absA12);

    if absA13 > max_off_diag {
        (p, q, max_off_diag) = (0, 2, absA13);
    }
    if absA23 > max_off_diag {
        (p, q, max_off_diag) = (1, 2, absA23);
    }

    (p, q, max_off_diag)
}

pub(crate) fn compute_two_sided_rotation<T: FloatT>(
    App: T,
    Aqp: T,
    Apq: T,
    Aqq: T,
) -> (T, T, T, T, T, T) {
    // polar rotation to symmetrize
    let (cp, sp, S11, S12, S22) = compute_polar_2x2(App, Aqp, Apq, Aqq);

    // then symmetric rotation to diagonalize S
    let (cj, sj, t) = compute_jacobi_rotation(S12, S11, S22);

    // the RHS rotation is the jacobi one
    let (cr, sr) = (cj, sj);

    // the diagonal elements
    let d1 = S11 - t * S12;
    let d2 = S22 + t * S12;

    // The LHS is the composition of the two rotations
    let cl = cp * cj - sp * sj;
    let sl = cp * sj + sp * cj;

    (cl, sl, cr, sr, d1, d2)
}

#[allow(clippy::too_many_arguments)]
fn apply_two_sided_rotation<T: FloatT>(
    A: &mut DenseMatrix3<T>,
    p: usize,
    q: usize,
    cl: T,
    sl: T,
    cr: T,
    sr: T,
    d1: T,
    d2: T,
) {
    debug_assert!(p < q);
    A[(p, p)] = d1;
    A[(q, q)] = d2;

    #[inline(always)]
    fn lr_rotate<T: FloatT>(
        A: &mut DenseMatrix3<T>,
        p: usize,
        q: usize,
        r: usize,
        cl: T,
        sl: T,
        cr: T,
        sr: T,
    ) {
        let (Apr, Aqr) = (A[(p, r)], A[(q, r)]);
        let (Arp, Arq) = (A[(r, p)], A[(r, q)]);
        A[(p, r)] = Apr * cl - Aqr * sl;
        A[(q, r)] = Aqr * cl + Apr * sl;
        A[(r, p)] = Arp * cr - Arq * sr;
        A[(r, q)] = Arq * cr + Arp * sr;
    }

    if p == 0 {
        if q == 1 {
            lr_rotate(A, 0, 1, 2, cl, sl, cr, sr);
        } else {
            lr_rotate(A, 0, 2, 1, cl, sl, cr, sr);
        }
    } else {
        lr_rotate(A, 1, 2, 0, cl, sl, cr, sr);
    }

    // eliminate the rotated off-diagonal
    A[(p, q)] = T::zero();
    A[(q, p)] = T::zero();
}

#[inline]
fn hypot_fast<T: FloatT>(x: T, y: T) -> T {

    // NB: avoids overflow in x^2 + y^2, and also faster 
    // bc it reduces the range of the sqrt to [1,2]
    // marginally faster than [x,y].norm(), which is 
    // equivalent and amounts to the same method. Faster 
    // than hypot(x,y) since it doesn't check nan or inf 
    // edge cases

    let x = x.abs();
    let y = y.abs();

    let (maxval, minval) = if x > y {
        (x, y)
    } else {
        (y, x)
    };

    if minval.is_zero() {
         if maxval.is_zero() {
            return T::zero();
        } else {
            return maxval;
        }
    }

    let r = minval / maxval;
    maxval * T::sqrt(T::one() + r * r)

}

fn compute_polar_2x2<T: FloatT>(App: T, Aqp: T, Apq: T, Aqq: T) -> (T, T, T, T, T) {
    // computes a rotation such that Q'A is symmetric

    let x = App + Aqq;
    let y = Aqp - Apq;

    let d = hypot_fast(x, y); 

    let (c, s) = {
        if d.is_zero() {
            (T::one(), T::zero())
        } else {
            (x / d, -y / d)
        }
    };

    let S11 = App * c - Aqp * s;
    let S22 = Aqq * c + Apq * s;
    let S12 = Apq * c - Aqq * s;

    (c, s, S11, S12, S22)
}

fn fix_order_and_signs_3x3<T: FloatT>(
    a: T,
    b: T,
    c: T,
    U: &mut DenseMatrix3<T>,
    V: &mut DenseMatrix3<T>,
) -> [T; 3] {
    // sort in descending order by absolute value

    #[inline(always)]
    fn abs_and_signbit<T: FloatT>(x: T) -> (T, bool) {
        (x.abs(), x.is_sign_negative())
    }

    let (mut absa, nega) = abs_and_signbit(a);
    let (mut absb, negb) = abs_and_signbit(b);
    let (mut absc, negc) = abs_and_signbit(c);

    // flip column signs first before sorting
    if nega {
        U.flip_column(0);
    }
    if negb {
        U.flip_column(1);
    }
    if negc {
        U.flip_column(2);
    }

    // Compare and swap a and b if needed
    if absa < absb {
        (absa, absb) = (absb, absa);
        U.swap_columns(0, 1);
        V.swap_columns(0, 1);
    }

    // Compare and swap b and c if needed
    if absa < absc {
        (absa, absc) = (absc, absa);
        U.swap_columns(0, 2);
        V.swap_columns(0, 2);
    }

    // One more swap to fix a and b only if necessary
    if absb < absc {
        (absb, absc) = (absc, absb);
        U.swap_columns(1, 2);
        V.swap_columns(1, 2);
    }

    [absa, absb, absc]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_3x3_nice() {
        #[rustfmt::skip]
        let A = Matrix::from(&[
            [1.0, 2.0, 3.0], 
            [4.0, 0.0, 2.0], 
            [-2.0, -8.0, -4.0]]
        );

        let strue = [9.970195528775319, 4.087406544070054, 1.374157509706204];

        let mut A: DenseMatrix3<f64> = A.into();
        let mut U = DenseMatrix3::zeros();
        let mut V = DenseMatrix3::zeros();

        // A solves in place, so make a copy
        let Aorig = A.clone();

        let s = A.svd(&mut U, &mut V);

        for i in 0..3 {
            assert!((s[i] - strue[i]).abs() < 1e-10);
        }

        for (i, &si) in s.iter().enumerate() {
            let mut u = U.col_slice(i).to_vec();
            let v = V.col_slice(i).to_vec();
            let mut Av = vec![0.0; v.len()];
            Aorig.mul(&mut Av, &v);
            u.scale(si);
            assert!(Av.norm_inf_diff(&u) < 1e-10);
        }
    }

    #[test]
    fn test_svd_3x3_hard() {
        let mut A = Matrix::zeros((3, 3));
        A.data.copy_from(&[0.0001, 0.01, 1.0, 0.01, 1.0, 0.0001, 1.0, 0.0001, 0.01]);

        let strue = [1.0101,0.9949869396127771, 0.9949869396127768];

        let mut A: DenseMatrix3<f64> = A.into();
        let mut U = DenseMatrix3::zeros();
        let mut V = DenseMatrix3::zeros();

        // A solves in place, so make a copy
        let Aorig = A.clone();

        let s = A.svd(&mut U, &mut V);

        for i in 0..3 {
            assert!((s[i] - strue[i]).abs() < 1e-10);
        }

        for (i, &si) in s.iter().enumerate() {
            let mut u = U.col_slice(i).to_vec();
            let v = V.col_slice(i).to_vec();
            let mut Av = vec![0.0; v.len()];
            Aorig.mul(&mut Av, &v);
            u.scale(si);
            assert!(Av.norm_inf_diff(&u) < 1e-10);
        }
    }
}

#[cfg(all(test, feature = "bench"))]
mod bench {

    use super::*;
    use itertools::iproduct;

    fn svd3_test_iter(fcn: fn(f64) -> f64) -> impl Iterator<Item = DenseMatrix3<f64>> {
        let v = [-4., -2., 0., 1., 5.];

        iproduct!(v, v, v, v, v, v, v, v, v).map(move |(a, b, c, d, e, f, g, h, i)| {
            let data = [
                fcn(a),
                fcn(b),
                fcn(c),
                fcn(d),
                fcn(e),
                fcn(f),
                fcn(g),
                fcn(h),
                fcn(i),
            ];
            DenseMatrix3 { data }
        })
    }

    fn bench_svd_grid_with_transform(fcn: fn(f64) -> f64, name: &str) {
        for A in svd3_test_iter(fcn) {
            let mut As = A.clone();
            let mut U = DenseMatrix3::zeros();
            let mut V = DenseMatrix3::zeros();

            let s = As.svd(&mut U, &mut V);

            // Check singular values
            assert!(s.iter().rev().is_sorted());

            //reconstruct matrix from SVD
            let mut Us = U.clone();
            for c in 0..s.len() {
                for r in 0..Us.nrows() {
                    Us[(r, c)] *= s[c];
                }
            }

            V.transpose_in_place();

            let mut M = Matrix::<f64>::zeros((3, 3));
            M.mul(&Us, &V, 1., 0.);

            if M.data().norm_inf_diff(A.data()) > fcn((1e-10).as_T()) {
                panic!("SVD reconstruction failed for {}:  A = {:?}", name, A);
            }
        }
    }

    #[test]
    fn bench_svd_grid() {
        // Linear transformation: f(x) = x
        bench_svd_grid_with_transform(|x| x, "linear");
    }

    #[test]
    fn bench_svd_exp_grid() {
        // Exponential transformation: f(x) = 10^x
        bench_svd_grid_with_transform(|x| (10_f64).powf(x), "10^x");
    }
}
