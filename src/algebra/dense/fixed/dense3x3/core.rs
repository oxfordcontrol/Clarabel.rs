#![allow(non_snake_case)]

use crate::algebra::*;

// 3x3 Dense matrix types are restricted to the crate
// NB: Implements square and symmetric 3x3 types to support
// power and exponential cones, plus special methods
// for 3x3 matrix decompositions.

// NB: S = 9 here because the matrix has 3^2 elements
#[allow(dead_code)] // used in tests even with `sdp` not selected
pub(crate) type DenseMatrix3<T> = DenseMatrixN<9, T>;

// NB: S = 6 here because the upper triangle has 6 elements
pub(crate) type DenseMatrixSym3<T> = DenseMatrixSymN<6, T>;

// hand implemented DenseMatrixSym3<T> to make sure
// everything is properly unrolled

impl<T> DenseMatrixSym3<T>
where
    T: FloatT,
{
    // y = H*x
    pub fn mul(&self, y: &mut [T], x: &[T]) {
        let H = self;

        //matrix is packed triu of a 3x3, so unroll it here
        y[0] = (H.data[0] * x[0]) + (H.data[1] * x[1]) + (H.data[3] * x[2]);
        y[1] = (H.data[1] * x[0]) + (H.data[2] * x[1]) + (H.data[4] * x[2]);
        y[2] = (H.data[3] * x[0]) + (H.data[4] * x[1]) + (H.data[5] * x[2]);
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
}

// internal unit tests
#[test]
fn test_3x3_sym_matrix_index() {
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
fn test_3x3_sym_index_linear() {
    let H = DenseMatrixSym3::<f64>::zeros();
    // upper triangular mapping
    assert_eq!(H.index_linear((0, 0)), 0);
    assert_eq!(H.index_linear((0, 1)), 1);
    assert_eq!(H.index_linear((1, 1)), 2);
    assert_eq!(H.index_linear((0, 2)), 3);
    assert_eq!(H.index_linear((1, 2)), 4);
    assert_eq!(H.index_linear((2, 2)), 5);

    // and the implied lower parts
    assert_eq!(H.index_linear((1, 0)), 1);
    assert_eq!(H.index_linear((2, 0)), 3);
    assert_eq!(H.index_linear((2, 1)), 4);
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

#[test]
fn test_3x3_transpose_in_place() {
    let H1 = Matrix::from(&[[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]]);

    let H2 = Matrix::from(&[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]);

    let mut A = DenseMatrix3::from(H1);
    A.transpose_in_place();
    assert_eq!(A.data(), H2.data());
}

#[test]
fn test_3x3_from_sym() {
    #[rustfmt::skip]
    let H = Matrix::from(&[
        [1., 4., 7.],
        [2., 5., 8.],
        [3., 6., 9.],
    ]);

    let A = DenseMatrixSym3::from(H.clone().sym_up());
    assert_eq!(A.data(), [1., 4., 5., 7., 8., 9.]);

    let A = DenseMatrixSym3::from(H.clone().sym_lo());
    assert_eq!(A.data(), [1., 2., 5., 3., 6., 9.]);
}

#[test]
fn test_3x3_sym_into_matrix() {
    #[rustfmt::skip]
    let A = Matrix::from(&[
        [1., 2., 3.],
        [2., 5., 8.],
        [3., 8., 9.],
    ]);

    let B = DenseMatrixSym3 {
        data: [1., 2., 5., 3., 8., 9.],
    };

    let B: Matrix<f64> = B.into();

    assert_eq!(A.data(), B.data());
}
