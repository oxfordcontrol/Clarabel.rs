use crate::algebra::*;
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

impl<'a, T> DenseMatrix for ReshapedMatrix<'a, T>
where
    T: FloatT,
{
    type T = T;
    fn index_linear(&self, idx: (usize, usize)) -> usize {
        idx.0 + self.m * idx.1
    }
    fn data(&self) -> &[T] {
        self.data
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

    /// Returns a Symmetric view of a triu matrix
    pub fn sym(&self) -> Symmetric<'_, Self> {
        debug_assert!(self.is_triu());
        Symmetric { src: self }
    }

    /// Set A = (A + A') / 2.  Assumes A is real
    pub fn symmetric_part(&mut self) -> &mut Self {
        assert!(self.is_square());
        let half: T = (0.5_f64).as_T();

        for r in 0..self.m {
            for c in 0..r - 1 {
                let val = half * (self[(r, c)] + self[(c, r)]);
                self[(c, r)] = val;
                self[(r, c)] = val;
            }
        }
        self
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
        for c in 0..self.ncols() {
            for r in (c + 1)..self.nrows() {
                if self[(r, c)] != T::zero() {
                    return false;
                }
            }
        }
        true
    }
}

impl<'a, T> ReshapedMatrix<'a, T>
where
    T: FloatT,
{
    pub fn from_slice(data: &'a [T], m: usize, n: usize) -> Self {
        Self { data, m, n }
    }

    pub fn reshape(&mut self, size: (usize, usize)) -> &Self {
        assert!(size.0 * size.1 == self.m * self.n);
        self.m = size.0;
        self.n = size.1;
        self
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

macro_rules! impl_mat_index {
    ($mat:ty) => {
        impl<T: FloatT> Index<(usize, usize)> for $mat {
            type Output = T;
            fn index(&self, idx: (usize, usize)) -> &Self::Output {
                &self.data()[self.index_linear(idx)]
            }
        }
    };
}
impl_mat_index!(Matrix<T>);
impl_mat_index!(ReshapedMatrix<'_, T>);
impl_mat_index!(Adjoint<'_, Matrix<T>>);

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
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
}

impl<T> Symmetric<'_, Matrix<T>>
where
    T: FloatT,
{
    pub(crate) fn pack_triu(&self, v: &mut [T]) {
        let n = self.ncols();
        let numel = (n * (n + 1)) >> 1;
        assert!(v.len() == numel);

        let mut k = 0;
        for col in 0..self.ncols() {
            for row in 0..col {
                v[k] = self.src[(row, col)];
                k += 1;
            }
        }
    }
}

impl<T> std::fmt::Display for Matrix<T>
where
    T: FloatT,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        _display_matrix(self, f)
    }
}

fn _display_matrix<M>(m: &M, f: &mut std::fmt::Formatter) -> std::fmt::Result
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

#[test]
fn test_matrix_istriu() {
    use crate::algebra::Matrix;
    let (m, n) = (3, 3);
    let a = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0];
    let mat = Matrix::new_from_slice((m, n), &a);
    assert!(mat.is_triu());
}
