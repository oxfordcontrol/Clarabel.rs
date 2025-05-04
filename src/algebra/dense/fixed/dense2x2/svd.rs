#![allow(non_snake_case)]
use crate::algebra::*;
use crate::algebra::dense::fixed::dense3x3::svd::compute_two_sided_rotation;

// 2x2 SVD using a two-sided Jacobi method.

impl<T> DenseMatrix2<T>
where
    T: FloatT,
{
    pub(crate) fn svd(&mut self, U: &mut DenseMatrix2<T>, V: &mut DenseMatrix2<T>) -> [T; 2] {
        // two-sided Jacobi method for SVD

        let A = self;
        // in this case we can do the SVD in one shot
        // b/c a single givens + Jacobi rotation is enough
        let (cl, sl, cr, sr, d1, d2) =
            compute_two_sided_rotation(A[(0, 0)], A[(1, 0)], A[(0, 1)], A[(1, 1)]);

        set_order_and_signs_2x2(cl, sl, cr, sr, d1, d2, U, V)
    }
}

#[allow(clippy::too_many_arguments)]
fn set_order_and_signs_2x2<T: FloatT>(
    cl: T,
    sl: T,
    cr: T,
    sr: T,
    d1: T,
    d2: T,
    U: &mut DenseMatrix2<T>,
    V: &mut DenseMatrix2<T>,
) -> [T; 2] {
    // sort in descending order by absolute value
    #[inline(always)]
    fn abs_and_signbit<T: FloatT>(x: T) -> (T, bool) {
        (x.abs(), x.is_sign_negative())
    }
    let (mut absa, nega) = abs_and_signbit(d1);
    let (mut absb, negb) = abs_and_signbit(d2);

    U[(0, 0)] = cl;
    U[(1, 0)] = -sl;
    U[(0, 1)] = sl;
    U[(1, 1)] = cl;

    V[(0, 0)] = cr;
    V[(1, 0)] = -sr;
    V[(0, 1)] = sr;
    V[(1, 1)] = cr;

    // flip column signs first before sorting
    if nega {
        U.flip_column(0);
    }
    if negb {
        U.flip_column(1);
    }

    // Compare and swap a and b if needed
    if absa < absb {
        (absa, absb) = (absb, absa);
        U.swap_columns(0, 1);
        V.swap_columns(0, 1);
    }

    [absa, absb]
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svd_2x2_nice(){
        
        #[rustfmt::skip]
        let A = Matrix::from(&[
            [ 1.0, 2.0], 
            [ 3.0, 4.0]
        ]);
        let strue = [
            5.464985704219043,
            0.3659661906262575];

        let mut A: DenseMatrix2<f64> = A.into();
        let mut U = DenseMatrix2::zeros();
        let mut V = DenseMatrix2::zeros();

        let s = A.svd(&mut U, &mut V);

        for i in 0..2 {
            assert!((s[i] - strue[i]).abs() < 1e-10);
        }

        for (i,&si) in s.iter().enumerate() {
                let mut u = U.col_slice(i).to_vec();
                let v = V.col_slice(i).to_vec();
                let mut Av = vec![0.0; v.len()];
                A.mul(&mut Av, &v);
                u.scale(si);
                assert!(Av.norm_inf_diff(&u) < 1e-10);
        }
    }
}

