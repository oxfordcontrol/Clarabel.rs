#![allow(non_snake_case)]

use crate::algebra::*;
use std::ops::{Index, IndexMut};

// statically sized symmetric matrix types are restricted to the crate
//
// NB: Implements a symmetric type type to support
// power and exponential cones, plus special methods
// for 2x2 and 3x3 matrix decompositions.
//
// Data is stored as an array of N values belonging
// the upper triangle of a matrix, so N is implicitly
// required to be a triangular number.

#[derive(Debug, Clone)]
pub(crate) struct DenseMatrixSymN<const S: usize, T> {
    pub data: [T; S],
}

// stupid hack because I can't set S = N(N-1)/2
// for the array size in the type definition
impl<const S: usize, T: FloatT> DenseMatrixSymN<S, T> {
    pub const N: usize = Self::get_dimension();
    /// Compute N such that N(N-1)/2 = S
    const fn get_dimension() -> usize {
        let mut n = 0;
        while triangular_number(n) < S {
            n += 1;
        }
        assert!(triangular_number(n) == S);
        n
    }
    pub(crate) const fn dim(&self) -> usize {
        Self::N
    }
}

impl<const S: usize, T: FloatT> ShapedMatrix for DenseMatrixSymN<S, T> {
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
    fn size(&self) -> (usize, usize) {
        (Self::N, Self::N)
    }
}

impl<const S: usize, T: FloatT> DenseMatrix<T> for DenseMatrixSymN<S, T> {
    //convert row col coordinate to triu index
    #[inline]
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        let (r, c) = idx;
        if r < c {
            r + triangular_number(c)
        } else {
            c + triangular_number(r)
        }
    }
    #[inline]
    fn data(&self) -> &[T] {
        &self.data
    }
}

impl<const S: usize, T: FloatT> Index<(usize, usize)> for DenseMatrixSymN<S, T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[self.index_linear(idx)]
    }
}

impl<const S: usize, T: FloatT> IndexMut<(usize, usize)> for DenseMatrixSymN<S, T> {
    #[inline]
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.index_linear(idx)]
    }
}

impl<const S: usize, T: FloatT> DenseMatrixMut<T> for DenseMatrixSymN<S, T> {
    #[inline]
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<const S: usize, T: FloatT> DenseMatrixSymN<S, T> {
    pub fn zeros() -> Self {
        Self {
            data: [T::zero(); S],
        }
    }

    pub fn scaled_from(&mut self, c: T, B: &Self) {
        for i in 0..S {
            self.data[i] = c * B.data[i];
        }
    }

    pub fn copy_from(&mut self, src: &Self) {
        self.data.copy_from_slice(&src.data);
    }
}

impl<const S: usize, T> From<DenseMatrixSymN<S, T>> for Matrix<T>
where
    T: FloatT,
{
    fn from(B: DenseMatrixSymN<S, T>) -> Self {
        let n: usize = B.dim();
        let mut A = Self::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                A[(i, j)] = B[(i, j)];
            }
        }
        A
    }
}

// convert from an NxN symmetric matrix to the packed
// upper triangle representation.

impl<const S: usize, M, T> From<Symmetric<'_, M>> for DenseMatrixSymN<S, T>
where
    T: FloatT,
    M: DenseMatrix<T>,
{
    fn from(B: Symmetric<'_, M>) -> Self {
        debug_assert!(
            B.src.size() == (Self::N, Self::N),
            "Matrix must be NxN to convert to DenseMatrixSymN",
        );

        let mut A = Self::zeros();

        match B.uplo {
            MatrixTriangle::Triu => {
                //read from upper triangle
                for c in 0..Self::N {
                    for r in 0..=c {
                        A[(r, c)] = B.src[(r, c)];
                    }
                }
            }
            MatrixTriangle::Tril => {
                //read from lower triangle
                for c in 0..Self::N {
                    for r in c..Self::N {
                        A[(r, c)] = B.src[(r, c)];
                    }
                }
            }
        }
        A
    }
}
