#![allow(non_snake_case)]
// some functions are only used with 3x3 eigen or svd
// decompositions, which are only used in the sdp feature
#![allow(dead_code)]

use crate::algebra::*;
use std::ops::{Index, IndexMut};

// statically sized square matrix types are restricted to the crate
//
// NB: Implements a static square type to support special methods
// for 2x2 and 3x3 matrix decompositions.
//
// Data is stored as an array of N = dim^2 values

#[derive(Debug, Clone)]
pub(crate) struct DenseMatrixN<const S: usize, T> {
    pub data: [T; S],
}

// stupid hack because I can't set S = N*N

impl<const S: usize, T: FloatT> DenseMatrixN<S, T> {
    pub const N: usize = Self::get_dimension();
    /// Compute N such that N * N = S
    const fn get_dimension() -> usize {
        let mut n = 0;
        while n * n < S {
            n += 1;
        }
        assert!(n * n == S);
        n
    }
    pub(crate) const fn dim(&self) -> usize {
        Self::N
    }

    // PJG: this probably wants to be defined in some
    // more general matrix trait.   Same for most/all
    // the other functions defined in this trait
    pub(crate) fn set_identity(&mut self) {
        self.data.fill(T::zero());
        for i in 0..Self::N {
            self[(i, i)] = T::one();
        }
    }

    #[allow(dead_code)] //used in tests.
    pub(crate) fn col_slice(&self, j: usize) -> &[T] {
        let start = j * Self::N;
        &self.data[start..(start + Self::N)]
    }

    pub(crate) fn swap_columns(&mut self, i: usize, j: usize) {
        let A = self;
        for r in 0..Self::N {
            (A[(r, i)], A[(r, j)]) = (A[(r, j)], A[(r, i)])
        }
    }

    pub(crate) fn flip_column(&mut self, i: usize) {
        let A = self;
        for r in 0..Self::N {
            A[(r, i)] = -A[(r, i)];
        }
    }

    pub(crate) fn transpose_in_place(&mut self) {
        for c in 0..Self::N {
            for r in 0..c {
                (self[(r, c)], self[(c, r)]) = (self[(c, r)], self[(r, c)]);
            }
        }
    }

    // duplicating the mul strait defined elsewhere
    // for now.  Possibly better for performance.
    #[allow(dead_code)] //used in tests.
    pub(crate) fn mul(&self, y: &mut [T], x: &[T]) {
        assert!(Self::N == x.len());
        assert!(Self::N == y.len());

        y.fill(T::zero());
        // column major order, but it probably doesn't matter
        for c in 0..Self::N {
            let xc = x[c];
            for r in 0..Self::N {
                y[r] += self[(r, c)] * xc;
            }
        }
    }
}

impl<const S: usize, T: FloatT> ShapedMatrix for DenseMatrixN<S, T> {
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
    fn size(&self) -> (usize, usize) {
        (Self::N, Self::N)
    }
}

impl<const S: usize, T: FloatT> DenseMatrix<T> for DenseMatrixN<S, T> {
    //convert row col coordinate to triu index
    #[inline]
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        idx.0 + idx.1 * Self::N
    }
    #[inline]
    fn data(&self) -> &[T] {
        &self.data
    }
}

impl<const S: usize, T: FloatT> Index<(usize, usize)> for DenseMatrixN<S, T> {
    type Output = T;
    #[inline]
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data[self.index_linear(idx)]
    }
}

impl<const S: usize, T: FloatT> IndexMut<(usize, usize)> for DenseMatrixN<S, T> {
    #[inline]
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        &mut self.data[self.index_linear(idx)]
    }
}

impl<const S: usize, T: FloatT> DenseMatrixMut<T> for DenseMatrixN<S, T> {
    #[inline]
    fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<const S: usize, T: FloatT> DenseMatrixN<S, T> {
    pub fn zeros() -> Self {
        Self {
            data: [T::zero(); S],
        }
    }
}

impl<const S: usize, T> From<DenseMatrixN<S, T>> for Matrix<T>
where
    T: FloatT,
{
    fn from(B: DenseMatrixN<S, T>) -> Self {
        let n: usize = B.dim();
        Self::new((n, n), B.data.to_vec())
    }
}

impl<const S: usize, T, S2> From<&DenseStorageMatrix<S2, T>> for DenseMatrixN<S, T>
where
    T: FloatT,
    S2: AsRef<[T]>,
{
    fn from(B: &DenseStorageMatrix<S2, T>) -> Self {
        debug_assert!(B.size() == (Self::N, Self::N));
        let mut A = Self::zeros();
        A.data.copy_from_slice(B.data());
        A
    }
}

impl<const S: usize, T, S2> From<&mut DenseStorageMatrix<S2, T>> for DenseMatrixN<S, T>
where
    T: FloatT,
    S2: AsMut<[T]> + AsRef<[T]>,
{
    fn from(B: &mut DenseStorageMatrix<S2, T>) -> Self {
        Self::from(&*B) //drop the mutable borrow
    }
}

impl<const S: usize, T, S2> From<DenseStorageMatrix<S2, T>> for DenseMatrixN<S, T>
where
    T: FloatT,
    S2: AsMut<[T]> + AsRef<[T]>,
{
    fn from(B: DenseStorageMatrix<S2, T>) -> Self {
        Self::from(&B)
    }
}
