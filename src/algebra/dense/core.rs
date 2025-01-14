// allow dead code here since dense matrix and its supporting
// functionality could eventually become a public interface.
#![allow(dead_code)]
#![allow(non_snake_case)]

use crate::algebra::*;
use num_traits::Num;

// The comment below is not a docstring since it relies on the 
// type Matrix<T> which is not currently visible outside the crate.
// It can be restored if the dense Matrix type is made public.

// Dense matrix in column major format
//
// __Example usage__ : To construct the 3 x 3 matrix
// ```text
// A = [1.  3.  5.]
//     [2.  0.  6.]
//     [0.  4.  7.]
/// ```
//
// ```no_run
// use clarabel::algebra::Matrix;
//
// let A : Matrix<f64> = Matrix::new(
//    (3, 3),  //size as tuple
//    vec![1., 2., 0., 3., 0., 4., 5., 6., 7.]
//  );
//
// ```
//
// Creates a Matrix from a slice of arrays.
//
// Example:
// ```
// use clarabel::algebra::Matrix;
// let A = Matrix::from(
//      &[[1.0, 2.0],
//        [3.0, 0.0],
//        [0.0, 4.0]]);
// ```
//
#[allow(clippy::needless_range_loop)]
impl<'a, I, J, T> From<I> for Matrix<T>
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = &'a T>,
    T: FloatT,
{
    fn from(rows: I) -> Matrix<T> {
        let rows: Vec<Vec<T>> = rows
            .into_iter()
            .map(|r| r.into_iter().copied().collect())
            .collect();

        let m = rows.len();
        let n = rows.iter().map(|r| r.len()).next().unwrap_or(0);
        assert!(rows.iter().all(|r| r.len() == n));
        let nnz = m * n;

        let mut data = Vec::with_capacity(nnz);
        for c in 0..n {
            for r in 0..m {
                data.push(rows[r][c]);
            }
        }

        Self::new((m,n), data)
    }
}


// Constructors for dense matrices with owned data 

impl<T> Matrix<T>
where
    T: FloatT,
{
    pub fn zeros(size: (usize, usize)) -> Self {
        let data = vec![T::zero(); size.0 * size.1];
        Self::new(size, data)
    }

    pub fn identity(n: usize) -> Self {
        let mut mat = Matrix::zeros((n, n));
        mat.set_identity();
        mat
    }

    pub fn new(size: (usize, usize), data: Vec<T>) -> Self {
        assert!(size.0 * size.1 == data.len());
        Self{size, data, phantom: std::marker::PhantomData}
    }

    pub fn new_from_slice(size: (usize, usize), src: &[T]) -> Self {
        Self::new(size, src.to_vec())
    }

    /// Resize a matrix, preserving or expanding allocated
    /// space.   Values are not flushed but may be garbage.
    pub fn resize(&mut self, size: (usize, usize)) {
        self.size = size;
        self.data.resize(size.0 * size.1, T::zero());
    }
}

// Methods that required mutable access to the matrix

impl<S,T> DenseStorageMatrix<S,T>
where
    S: AsMut<[T]> + AsRef<[T]>,
    T: Sized + Num + Copy,
{
    pub fn set_identity(&mut self) {
        assert!(self.is_square());
        self.data_mut().fill(T::zero());
        for i in 0..self.ncols() {
            self[(i, i)] = T::one();
        }
    }

    pub fn copy_from_slice(&mut self, src: &[T]) {
        self.data_mut().copy_from_slice(src);
    }

    pub fn t(&self) -> Adjoint<'_, Self> {
        Adjoint { src: self }
    }

    /// Returns a Symmetric view of a triu matrix
    pub fn sym(&self) -> Symmetric<'_, Self> {
        debug_assert!(self.is_triu());
        Symmetric { src: self }
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

    /// self.subsagn(rows,cols,B) sets self[rows,cols] = B
    pub(crate) fn subsasgn<'a, RI, CI, MAT>(&mut self, rows: RI, cols: CI, source: &MAT)
    where
        RI: IntoIterator<Item = &'a usize> + Copy,
        CI: IntoIterator<Item = &'a usize>,
        MAT: DenseMatrix<T,Output = T>,
    {
        for (j, &col) in cols.into_iter().enumerate() {
            for (i, &row) in rows.into_iter().enumerate() {
                self[(row, col)] = source[(i, j)];
            }
        }
    }

    /// self.subsref(B,rows,cols) sets self = B[rows,cols]
    pub(crate) fn subsref<'a, RI, CI, MAT>(
        &mut self,
        source: &MAT,
        rows: RI,
        cols: CI,
    ) where
        RI: IntoIterator<Item = &'a usize> + Copy,
        CI: IntoIterator<Item = &'a usize>,
        MAT: DenseMatrix<T,Output = T>,
    {
        for (j, &col) in cols.into_iter().enumerate() {
            for (i, &row) in rows.into_iter().enumerate() {
                self[(i, j)] = source[(row, col)];
            }
        }
    }
}




impl<T> std::fmt::Display for Matrix<T>
where
    T: FloatT,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f)?;
        for i in 0..self.nrows() {
            write!(f, "[ ")?;
            for j in 0..self.ncols() {
                write!(f, " {:?}", self[(i, j)])?;
            }
            writeln!(f, "]")?;
        }
        writeln!(f)?;
        Ok(())
    }
}




#[test]
#[rustfmt::skip]
fn test_matrix_istriu() {
    
    let A = Matrix::from(&[
        [1., 2., 3.], 
        [0., 2., 0.], 
        [0., 0., 1.]]); 

    assert!(A.is_triu());

    let A = Matrix::from(&[
        [1., 2., 3.], 
        [0., 2., 0.], 
        [1., 0., 1.]]); 

    assert!(!A.is_triu());
}

#[test]
#[rustfmt::skip]
fn test_matrix_from_slice_of_arrays() {
    let A = Matrix::new(
        (3, 2), // n
        vec![1., 3., 0., 2., 0., 4.],
    );

    let B = Matrix::from(&[
        [1., 2.], 
        [3., 0.], 
        [0., 4.]]); 

    let C: Matrix<f64> = (&[
        [1., 2.], 
        [3., 0.], 
        [0., 4.],
    ]).into();

    assert_eq!(A, B);
    assert_eq!(A, C);
}

#[test]
fn test_svec_conversions() {
    let n = 3;

    let X = Matrix::from(&[
        [1., 3., -2.], //
        [3., -4., 7.], //
        [-2., 7., 5.], //
    ]);

    let Y = Matrix::from(&[
        [2., 5., -4.],  //
        [5., 6., 2.],   //
        [-4., 2., -3.], //
    ]);

    let mut Z = Matrix::zeros((3, 3));

    let mut x = vec![0.; triangular_number(n)];
    let mut y = vec![0.; triangular_number(n)];

    // check inner product identity
    mat_to_svec(&mut x, &X);
    mat_to_svec(&mut y, &Y);

    assert!(f64::abs(x.dot(&y) - X.data().dot(Y.data())) < 1e-12);

    // check round trip
    mat_to_svec(&mut x, &X);
    svec_to_mat(&mut Z, &x);
    assert!(X.data().norm_inf_diff(Z.data()) < 1e-12);
}

#[test]
fn test_matrix_subsref() {
    let A = Matrix::from(&[
        [1., 4., 7.], //
        [2., 5., 8.], //
        [3., 6., 9.],
    ]);

    let Aperm = Matrix::from(&[
        [8., 5., 2.], //
        [7., 4., 1.],
        [9., 6., 3.],
    ]);

    let Ared = Matrix::from(&[
        [8., 2.], //
        [9., 3.],
    ]);

    let Ared2 = Matrix::from(&[
        [6., 4.], //
        [9., 7.],
    ]);

    let Aexpanded = Matrix::from(&[
        [8., 0., 2.], //
        [0., 0., 0.], //
        [9., 0., 3.],
    ]);

    let mut B = Matrix::zeros((3, 3));
    let rows: Vec<usize> = vec![1, 0, 2];
    let cols: Vec<usize> = vec![2, 1, 0];
    B.subsref(&A, &rows, &cols);

    assert_eq!(B, Aperm);

    let mut C = Matrix::zeros((2, 2));
    let rows: Vec<usize> = vec![1, 2];
    let cols: Vec<usize> = vec![2, 0];
    C.subsref(&A.t(), &rows, &cols);

    assert_eq!(C, Ared2);

    let mut D = Matrix::zeros((3, 3));
    D.subsasgn(&[0, 2], &[0, 2], &Ared);
    assert_eq!(D, Aexpanded);
}
