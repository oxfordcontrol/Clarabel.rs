use crate::algebra::*;
use std::ops::{Index, IndexMut};

// core dense matrix type for owned and borrowed matrices
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DenseStorageMatrix<S, T>
where
    S: AsRef<[T]>,
    T: Sized,
{
    /// dimensions
    pub size: (usize, usize),
    /// vector of data in column major format
    pub data: S,
    pub(crate) phantom: std::marker::PhantomData<T>,
}

pub(crate) type Matrix<T> = DenseStorageMatrix<Vec<T>, T>;
pub(crate) type BorrowedMatrix<'a, T> = DenseStorageMatrix<&'a [T], T>;
pub(crate) type BorrowedMatrixMut<'a, T> = DenseStorageMatrix<&'a mut [T], T>;

impl<S, T> ShapedMatrix for DenseStorageMatrix<S, T>
where
    S: AsRef<[T]>,
{
    fn size(&self) -> (usize, usize) {
        self.size
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
}

//NB: the concrete dense type is just called "Matrix".  The "DenseMatrix" trait
//is implemented on Matrix, Adjoint and BorrowedMatrix types to allow for indexing
//of values in any of those formats.   This follows the Julia naming convention
//for similar types.
pub(crate) trait DenseMatrix<T>: ShapedMatrix + Index<(usize, usize), Output = T> {
    fn index_linear(&self, idx: (usize, usize)) -> usize;
    fn data(&self) -> &[T];
}

pub(crate) trait DenseMatrixMut<T>: DenseMatrix<T> {
    fn data_mut(&mut self) -> &mut [T];
}

impl<S, T> DenseMatrix<T> for DenseStorageMatrix<S, T>
where
    S: AsRef<[T]>,
{
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        idx.0 + self.nrows() * idx.1
    }
    fn data(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<S, T> DenseMatrixMut<T> for DenseStorageMatrix<S, T>
where
    S: AsMut<[T]> + AsRef<[T]>,
{
    fn data_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

impl<S, T> Index<(usize, usize)> for DenseStorageMatrix<S, T>
where
    S: AsRef<[T]>,
    T: Sized,
{
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &T {
        let lidx = self.index_linear(idx);
        &self.data()[lidx]
    }
}

impl<S, T> IndexMut<(usize, usize)> for DenseStorageMatrix<S, T>
where
    S: AsRef<[T]> + AsMut<[T]>,
    T: Sized,
{
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        let lidx = self.index_linear(idx);
        &mut self.data_mut()[lidx]
    }
}

impl<S, T> DenseStorageMatrix<S, T>
where
    S: AsRef<[T]>,
{
    pub fn col_slice(&self, col: usize) -> &[T] {
        let (m, n) = self.size;
        assert!(col < n);
        &self.data()[(col * m)..(col + 1) * m]
    }
}

impl<S, T> DenseStorageMatrix<S, T>
where
    S: AsMut<[T]> + AsRef<[T]>,
{
    pub fn col_slice_mut(&mut self, col: usize) -> &mut [T] {
        let (m, n) = self.size;
        assert!(col < n);
        &mut self.data_mut()[(col * m)..(col + 1) * m]
    }
}

// ------------------------------------------------
// Adjoint and Symmetric implementations for DenseMatrix.
// These are read only views of the matrix that allow for
// things like matrix multiplication and indexing, but
// do not allow for modification of the underlying data.

impl<S, T> DenseMatrix<T> for Adjoint<'_, DenseStorageMatrix<S, T>>
where
    S: AsRef<[T]>,
    T: Sized,
{
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        //reverse the indices
        self.src.index_linear((idx.1, idx.0))
    }
    fn data(&self) -> &[T] {
        self.src.data()
    }
}

impl<S, T> DenseMatrix<T> for Symmetric<'_, DenseStorageMatrix<S, T>>
where
    S: AsRef<[T]>,
    T: Sized,
{
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        if idx.0 <= idx.1 {
            //triu part
            self.src.index_linear((idx.0, idx.1))
        } else {
            //tril part uses triu entry
            self.src.index_linear((idx.1, idx.0))
        }
    }
    fn data(&self) -> &[T] {
        self.src.data()
    }
}

impl<S, T> Index<(usize, usize)> for Adjoint<'_, DenseStorageMatrix<S, T>>
where
    S: AsRef<[T]>,
{
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &T {
        let lidx = self.index_linear(idx);
        &self.data()[lidx]
    }
}

impl<S, T> Index<(usize, usize)> for Symmetric<'_, DenseStorageMatrix<S, T>>
where
    S: AsRef<[T]>,
{
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &T {
        let lidx = self.index_linear(idx);
        &self.data()[lidx]
    }
}

impl<T> Symmetric<'_, Matrix<T>>
where
    T: FloatT,
{
    pub(crate) fn pack_triu(&self, v: &mut [T]) {
        let n = self.ncols();
        let numel = triangular_number(n);
        assert!(v.len() == numel);

        let mut k = 0;
        for col in 0..self.ncols() {
            for row in 0..=col {
                v[k] = self.src[(row, col)];
                k += 1;
            }
        }
    }
}
