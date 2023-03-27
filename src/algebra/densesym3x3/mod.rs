#![allow(non_snake_case)]

use super::FloatT;
use crate::algebra::*;
use std::ops::{Index, IndexMut};

// 3x3 Dense matrix types are restricted to the crate
// NB: Implements a symmetric 3x3 type to support
// power and exponential cones.
//
// Data is stored as an array of 6 values belonging
// the upper triangle of a 3x3 matrix.   Lower triangle
// is assumed symmetric.  Any read/write to the lower
// triangle is handled by reversing the indices

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenseMatrixSym3<T> {
    pub data: [T; 6],
}

impl<T: FloatT> Index<(usize, usize)> for DenseMatrixSym3<T> {
    type Output = T;

    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[Self::index_linear(idx)]
    }
}

impl<T: FloatT> IndexMut<(usize, usize)> for DenseMatrixSym3<T> {
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.data[Self::index_linear(idx)]
    }
}

impl<T> DenseMatrixSym3<T>
where
    T: FloatT,
{
    pub fn zeros() -> Self {
        Self {
            data: [T::zero(); 6],
        }
    }

    // y = H*x
    pub fn mul(&self, y: &mut [T], x: &[T]) {
        let H = self;

        //matrix is packed triu of a 3x3, so unroll it here
        y[0] = (H.data[0] * x[0]) + (H.data[1] * x[1]) + (H.data[3] * x[2]);
        y[1] = (H.data[1] * x[0]) + (H.data[2] * x[1]) + (H.data[4] * x[2]);
        y[2] = (H.data[3] * x[0]) + (H.data[4] * x[1]) + (H.data[5] * x[2]);
    }

    pub fn scaled_from(&mut self, c: T, B: &Self) {
        for i in 0..6 {
            self.data[i] = c * B.data[i];
        }
    }

    pub fn norm_fro(&self) -> T {
        let d = self.data;
        //Frobenius norm.   Need to be careful to count
        //the packed off diagonals twice
        let mut sumsq = T::zero();

        //diagonal terms
        sumsq += d[0] * d[0] + d[2] * d[2] + d[5] * d[5];
        //off diagonals
        sumsq += (d[1] * d[1] + d[3] * d[3] + d[4] * d[4]) * (2.).as_T();

        sumsq.sqrt()
    }

    //returns y'*H*x
    pub fn quad_form(&self, y: &[T], x: &[T]) -> T {
        let H = self;
        let mut out = T::zero();
        //matrix is packed 3x3, so unroll it here
        out += y[0] * (H.data[0] * x[0] + H.data[1] * x[1] + H.data[3] * x[2]);
        out += y[1] * (H.data[1] * x[0] + H.data[2] * x[1] + H.data[4] * x[2]);
        out += y[2] * (H.data[3] * x[0] + H.data[4] * x[1] + H.data[5] * x[2]);
        out
    }

    pub fn copy_from(&mut self, src: &Self) {
        self.data.copy_from_slice(&src.data);
    }

    //convert row col coordinate to triu index
    #[inline]
    pub fn index_linear(idx: (usize, usize)) -> usize {
        let (r, c) = idx;
        if r < c {
            r + triangular_number(c)
        } else {
            c + triangular_number(r)
        }
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
    //  This might be faster if the double indexing were also
    //  unrolled to the underlying linear index, but it would
    //  then be even more unreadable and probably not worth it.

    pub fn cholesky_3x3_explicit_factor(&mut self, A: &DenseMatrixSym3<T>) -> bool {
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

// internal unit tests
#[test]
fn test_3x3_matrix_index() {
    let mut H = DenseMatrixSym3::zeros();

    // assume upper triangle, check lower
    H[(0, 0)] = 1.;
    H[(0, 1)] = 2.;
    H[(1, 1)] = 3.;
    H[(0, 2)] = 4.;
    H[(1, 2)] = 5.;
    H[(2, 2)] = 6.;

    assert_eq!(H[(1, 0)], 2.);
    assert_eq!(H[(2, 0)], 4.);
    assert_eq!(H[(2, 1)], 5.);

    // data should be packed like this
    let data = [1., 2., 3., 4., 5., 6.];

    assert!(
        std::iter::zip(H.data, data).all(|(a, b)| a == b),
        "Arrays are not equal"
    );

    assert_eq!(H[(0, 1)], H[(1, 0)]);
    assert_eq!(H[(0, 2)], H[(2, 0)]);
    assert_eq!(H[(2, 1)], H[(1, 2)]);
}

#[test]
fn test_3x3_index_linear() {
    // upper triangular mapping
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((0, 0)), 0);
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((0, 1)), 1);
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((1, 1)), 2);
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((0, 2)), 3);
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((1, 2)), 4);
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((2, 2)), 5);

    // and the implied lower parts
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((1, 0)), 1);
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((2, 0)), 3);
    assert_eq!(DenseMatrixSym3::<f64>::index_linear((2, 1)), 4);
}

#[test]
fn test_3x3_ops() {
    let x = [-2., -7., 3.];
    let mut y = [0.; 3];

    let mut H = DenseMatrixSym3::<f64>::zeros();
    H[(0, 0)] = 1.;
    H[(0, 1)] = 2.;
    H[(1, 1)] = 3.;
    H[(0, 2)] = 4.;
    H[(1, 2)] = 5.;
    H[(2, 2)] = 6.;

    //multiplication
    H.mul(&mut y, &x);

    assert_eq!(y[0], -4.);
    assert_eq!(y[1], -10.);
    assert_eq!(y[2], -25.);

    //frobenius norm
    assert!(f64::abs(H.norm_fro() - 11.661903789690601) < 1e-15);

    //quadratic form
    assert_eq!(H.quad_form(&x, &x), 3.);
}
