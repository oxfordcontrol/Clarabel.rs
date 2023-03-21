use crate::algebra::{
    Adjoint, DenseMatrix, FloatT, Matrix, MatrixShape, ShapedMatrix, Symmetric, VectorMath,
};
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

    pub fn identity(n: usize) -> Self {
        let mut mat = Matrix::zeros((n, n));
        mat.set_identity();
        mat
    }

    pub fn set_identity(&mut self) {
        assert!(self.m == self.n);
        self.data_mut().set(T::zero());
        for i in 0..self.n {
            self[(i, i)] = T::one();
        }
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

    pub fn sym(&self) -> Symmetric<'_, Self> {
        debug_assert!(self.is_triu());
        Symmetric { src: self }
    }

    pub fn col_slice(&self, col: usize) -> &[T] {
        assert!(col < self.n);
        &self.data[(col * self.m)..(col + 1) * self.m]
    }

    pub fn col_slice_mut(&mut self, col: usize) -> &mut [T] {
        assert!(col < self.n);
        &mut self.data[(col * self.m)..(col + 1) * self.m]
    }

    pub fn is_triu(&self) -> bool {
        // check lower triangle for any nonzero entries
        for r in 0..self.nrows() {
            for c in (r + 1)..self.ncols() {
                if self[(r, c)] != T::zero() {
                    return false;
                }
            }
        }
        true
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
