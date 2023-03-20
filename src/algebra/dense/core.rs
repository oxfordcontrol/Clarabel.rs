use crate::algebra::{Adjoint, DenseMatrix, FloatT, Matrix, MatrixShape, ShapedMatrix};
use std::ops::{Index, IndexMut};

impl<T> DenseMatrix for Matrix<T>
where
    T: FloatT,
{
    type T = T;
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        idx.0 + self.m * idx.1
    }
    fn data(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T> DenseMatrix for Adjoint<'a, Matrix<T>>
where
    T: FloatT,
{
    type T = T;
    #[inline]
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        self.src.index_linear((idx.1, idx.0))
    }
    fn data(&self) -> &[T] {
        &self.src.data
    }
}

impl<T> Matrix<T>
where
    T: FloatT,
{
    pub fn zeros(size: (usize, usize)) -> Self {
        let (m, n) = size;
        let data = vec![T::zero(); m * n];

        Self { m, n, data }
    }

    pub fn new_from_slice(size: (usize, usize), src: &[T]) -> Self {
        let (m, n) = size;
        assert!(m * n == src.len());
        Self {
            m,
            n,
            data: src.to_vec(),
        }
    }

    pub fn copy_from_slice(&mut self, src: &[T]) -> &mut Self {
        self.data.copy_from_slice(src);
        self
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn t(&self) -> Adjoint<'_, Self> {
        Adjoint { src: self }
    }

    pub fn col_slice(&self, col: usize) -> &[T] {
        assert!(col < self.n);
        &self.data[(col * self.m)..(col + 1) * self.m]
    }

    pub fn col_slice_mut(&mut self, col: usize) -> &mut [T] {
        assert!(col < self.n);
        &mut self.data[(col * self.m)..(col + 1) * self.m]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T>
where
    T: FloatT,
{
    fn index_mut(&mut self, idx: (usize, usize)) -> &mut Self::Output {
        let lidx = self.index_linear(idx);
        &mut self.data[lidx]
    }
}

impl<T> Index<(usize, usize)> for Matrix<T>
where
    T: FloatT,
{
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data()[self.index_linear(idx)]
    }
}

impl<T> Index<(usize, usize)> for Adjoint<'_, Matrix<T>>
where
    T: FloatT,
{
    type Output = T;
    fn index(&self, idx: (usize, usize)) -> &Self::Output {
        &self.data()[self.index_linear(idx)]
    }
}

impl<T> ShapedMatrix for Matrix<T>
where
    T: FloatT,
{
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
    fn is_square(&self) -> bool {
        self.nrows() == self.ncols()
    }
}

impl<'a, T> ShapedMatrix for Adjoint<'a, Matrix<T>>
where
    T: FloatT,
{
    fn nrows(&self) -> usize {
        self.src.ncols()
    }
    fn ncols(&self) -> usize {
        self.src.nrows()
    }
    fn size(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::T
    }
    fn is_square(&self) -> bool {
        self.src.is_square()
    }
}

impl<T> std::fmt::Display for Matrix<T>
where
    T: FloatT,
{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        display_matrix(self, f)
    }
}

fn display_matrix<M>(m: &M, f: &mut std::fmt::Formatter) -> std::fmt::Result
where
    M: DenseMatrix,
    M::Output: FloatT,
{
    writeln!(f)?;
    for i in 0..m.nrows() {
        write!(f, "[ ")?;
        for j in 0..m.ncols() {
            write!(f, " {:?}", m[(i, j)])?;
        }
        writeln!(f, "]")?;
    }
    writeln!(f)?;
    Ok(())
}
