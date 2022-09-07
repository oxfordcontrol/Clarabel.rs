#![allow(non_snake_case)]

use super::{FloatT, VectorMath};
use std::ops::{Index, IndexMut};

// Dense matrix types are restricted to the crate
// PJG: I will implement a basic 3x3 container here
// to support power and exponential cones, and then
// revisit it for SDPs.  We will want something that
// is compatible (interchangeably) with nalgebra /
// ndarray / some other blas like interface

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenseMatrix3<T> {
    pub data: [T; 9],
}

impl<T> Index<(usize, usize)> for DenseMatrix3<T> {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[idx.0 + 3 * idx.1]
    }
}

impl<T> IndexMut<(usize, usize)> for DenseMatrix3<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.data[idx.0 + 3 * idx.1]
    }
}

impl<T> DenseMatrix3<T>
where
    T: FloatT,
{
    pub fn zeros() -> Self {
        Self {
            data: [T::zero(); 9],
        }
    }

    pub fn quad_form(&self, y: &[T], x: &[T]) -> T {
        let mut out = T::zero();
        for i in 0..3 {
            for j in 0..3 {
                out += self[(i, j)] * y[i] * x[j];
            }
        }
        out
    }

    pub fn copy_from(&mut self, src: &Self) {
        self.data[..].copy_from(&src.data);
    }

    pub fn pack_triu(&self, v: &mut [T]) {
        //stores the upper triangle into a vector
        let d = self.data;
        assert_eq!(v.len(), 6);

        (v[0], v[1], v[2], v[3], v[4], v[5]) = (d[0], d[3], d[4], d[6], d[7], d[8]);
    }

    // ------------------------------
    //  special methods for solving 3x3 positive definite systems
    // ------------------------------

    //  Unrolled 3x3 cholesky decomposition without pivoting
    //  Returns `false` for a non-positive pivot and the
    //  factorization is not completed
    //
    //  NB: this is only marginally slower than the explicit
    //  3x3 LDL decomposition, which would avoid sqrts.

    pub fn cholesky_3x3_explicit_factor(&mut self, A: &DenseMatrix3<T>) -> bool {
        // PJG: This should return Result

        let L = self;

        let t = A[(0, 0)];
        if t <= T::zero() {
            return false;
        }

        L[(0, 0)] = t.sqrt();
        L[(1, 0)] = A[(1, 0)] / L[(0, 0)];

        let t = A[(1, 1)] - L[(1, 0)] * L[(1, 0)];

        if t <= T::zero() {
            return false;
        }

        L[(1, 1)] = t.sqrt();
        L[(2, 0)] = A[(2, 0)] / L[(0, 0)];
        L[(2, 1)] = (A[(2, 1)] - L[(1, 0)] * L[(2, 0)]) / L[(1, 1)];

        let t = A[(2, 2)] - L[(2, 0)] * L[(2, 0)] - L[(2, 1)] * L[(2, 1)];

        if t <= T::zero() {
            return false;
        }
        L[(2, 2)] = t.sqrt();

        true
    }

    // Unrolled 3x3 forward/backward substition for a Cholesky factor

    pub fn cholesky_3x3_explicit_solve(&self, x: &mut [T], b: &[T]) {
        let L = self;

        let c1 = b[0] / L[(0, 0)];
        let c2 = (b[1] * L[(0, 0)] - b[0] * L[(1, 0)]) / (L[(0, 0)] * L[(1, 1)]);
        let c3 = (b[2] * L[(0, 0)] * L[(1, 1)] - b[1] * L[(0, 0)] * L[(2, 1)]
            + b[0] * L[(1, 0)] * L[(2, 1)]
            - b[0] * L[(1, 1)] * L[(2, 0)])
            / (L[(0, 0)] * L[(1, 1)] * L[(2, 2)]);

        x[0] = (c1 * L[(1, 1)] * L[(2, 2)] - c2 * L[(1, 0)] * L[(2, 2)]
            + c3 * L[(1, 0)] * L[(2, 1)]
            - c3 * L[(1, 1)] * L[(2, 0)])
            / (L[(0, 0)] * L[(1, 1)] * L[(2, 2)]);
        x[1] = (c2 * L[(2, 2)] - c3 * L[(2, 1)]) / (L[(1, 1)] * L[(2, 2)]);
        x[2] = c3 / L[(2, 2)];
    }
}
