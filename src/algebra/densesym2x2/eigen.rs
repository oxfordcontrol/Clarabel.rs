#![allow(non_snake_case)]

// 2z2 symmetric eigendecomposition using Jacobi method.
// This uses Jacobi rotations from the 3x3 method, but
// only one rotation is required

use crate::algebra::{compute_jacobi_rotation, DenseMatrixMut, FloatT, Matrix};

use super::DenseMatrixSym2;

impl<T> DenseMatrixSym2<T>
where
    T: FloatT,
{
    pub(crate) fn eigvals(&mut self) -> [T; 2] {
        eig2(self, None)
    }

    pub(crate) fn eigen(&mut self, V: &mut Matrix<T>) -> [T; 2] {
        eig2(self, Some(V))
    }
}

fn eig2<T>(A: &DenseMatrixSym2<T>, V: Option<&mut Matrix<T>>) -> [T; 2]
where
    T: FloatT,
{
    // assume 2x2 input here
    let App = A.data[0];
    let Apq = A.data[1];
    let Aqq = A.data[2];
    let (c, s, t) = compute_jacobi_rotation(Apq, App, Aqq);
    // one step Givens rotation
    let tApq = t * Apq;
    let e1 = App - tApq;
    let e2 = Aqq + tApq;
    let noswap = e1 < e2;
    let e = if noswap { [e1, e2] } else { [e2, e1] };

    // compute eigenvectors if needed
    if let Some(V) = V {
        let Vp = V.data_mut();
        if noswap {
            Vp[0] = c;
            Vp[1] = -s;
            Vp[2] = s;
            Vp[3] = c;
        } else {
            Vp[0] = s;
            Vp[1] = c;
            Vp[2] = c;
            Vp[3] = -s;
        }
    }
    e
}

// ---- unit testing ----

#[cfg(test)]
mod test {

    use super::*;
    use crate::algebra::math_traits::VectorMath;
    use crate::algebra::DenseMatrixSym2;
    use itertools::iproduct;
    use std::ops::Range;

    fn sym2_test_iter(
        rng: Range<i32>,
        fcn: fn(f64) -> f64,
    ) -> impl Iterator<Item = DenseMatrixSym2<f64>> {
        iproduct!(rng.clone(), rng.clone(), rng.clone()).map(move |(a, b, c)| {
            let data = [
                fcn(a as f64), // Convert Item -> f64, then apply f
                fcn(b as f64),
                fcn(c as f64),
            ];
            DenseMatrixSym2 { data }
        })
    }

    fn check_eigvecs_with_tol(
        A: &DenseMatrixSym2<f64>,
        V: &Matrix<f64>,
        w: &[f64; 2],
        tol: f64,
    ) -> bool {
        let mut out = [0.; 2];

        for (i, wi) in w.iter().enumerate() {
            let mut v = V.col_slice(i).to_vec();
            A.mul(out.as_mut_slice(), &v);
            v.axpby(1.0, &out, -wi);
            if v.norm() > tol {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_eigen_grid() {
        let rng = -5..6;
        let iter = sym2_test_iter(rng, |x| x);
        let mut V = Matrix::zeros((2, 2));

        for As in iter {
            //this is the actual user facing method
            let mut Am = As.clone();
            let e = Am.eigen(&mut V);
            check_eigvecs_with_tol(&As, &V, &e, 1e-12);
        }
    }
}
