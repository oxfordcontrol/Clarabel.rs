#![allow(non_snake_case)]

use crate::algebra::{Adjoint, FloatT, MatrixShape, ShapedMatrix, SparseFormatError, Symmetric};
use std::iter::zip;

/// Sparse matrix in standard Compressed Sparse Column (CSC) format
///
/// __Example usage__ : To construct the 3 x 3 matrix
/// ```text
/// A = [1.  3.  5.]
///     [2.  0.  6.]
///     [0.  4.  7.]
/// ```
/// ```no_run
/// use clarabel::algebra::CscMatrix;
///
/// let A : CscMatrix<f64> = CscMatrix::new(
///    3,                                // m
///    3,                                // n
///    vec![0, 2, 4, 7],                 //colptr
///    vec![0, 1, 0, 2, 0, 1, 2],        //rowval
///    vec![1., 2., 3., 4., 5., 6., 7.], //nzval
///  );
///
/// // optional correctness check
/// assert!(A.check_format().is_ok());
///
/// // the above is equivalent to the following,
/// // which is more convenient for small matrices
/// let A = CscMatrix::from(
///      &[[1.0, 3.0, 5.0],
///        [2.0, 0.0, 6.0],
///        [0.0, 4.0, 7.0]]);
///
/// ```
///

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscMatrix<T = f64> {
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// CSC format column pointer.   
    ///
    /// Ths field should have length `n+1`. The last entry corresponds
    /// to the the number of nonzeros and should agree with the lengths
    /// of the `rowval` and `nzval` fields.
    pub colptr: Vec<usize>,
    /// vector of row indices
    pub rowval: Vec<usize>,
    /// vector of non-zero matrix elements
    pub nzval: Vec<T>,
}

/// Creates a CscMatrix from a slice of arrays.
///
/// Example:
/// ```
/// use clarabel::algebra::CscMatrix;
/// let A = CscMatrix::from(
///      &[[1.0, 2.0],
///        [3.0, 0.0],
///        [0.0, 4.0]]);
///
impl<'a, I, J, T> From<I> for CscMatrix<T>
where
    I: IntoIterator<Item = J>,
    J: IntoIterator<Item = &'a T>,
    T: FloatT,
{
    #[allow(clippy::needless_range_loop)]
    fn from(rows: I) -> CscMatrix<T> {
        let rows: Vec<Vec<T>> = rows
            .into_iter()
            .map(|r| r.into_iter().copied().collect())
            .collect();

        let m = rows.len();
        let n = rows.iter().map(|r| r.len()).next().unwrap_or(0);

        assert!(rows.iter().all(|r| r.len() == n));
        let nnz = rows.iter().flatten().filter(|&&v| v != T::zero()).count();

        let mut colptr = Vec::with_capacity(n + 1);
        let mut rowval = Vec::with_capacity(nnz);
        let mut nzval = Vec::<T>::with_capacity(nnz);

        colptr.push(0);
        for c in 0..n {
            for r in 0..m {
                let value = rows[r][c];
                if value != T::zero() {
                    rowval.push(r);
                    nzval.push(value);
                }
            }
            colptr.push(nzval.len());
        }

        CscMatrix::<T> {
            m,
            n,
            colptr,
            rowval,
            nzval,
        }
    }
}

impl<T> CscMatrix<T>
where
    T: FloatT,
{
    /// `CscMatrix` constructor.
    ///
    /// # Panics
    /// Makes rudimentary dimensional compatibility checks and panics on
    /// failure.   This constructor does __not__
    /// ensure that rows indices are all in bounds or that data is arranged
    /// such that entries within each column appear in order of increasing
    /// row index.   Responsibility for ensuring these conditions hold
    /// is left to the caller.
    ///

    pub fn new(m: usize, n: usize, colptr: Vec<usize>, rowval: Vec<usize>, nzval: Vec<T>) -> Self {
        assert_eq!(rowval.len(), nzval.len());
        assert_eq!(colptr.len(), n + 1);
        assert_eq!(colptr[n], rowval.len());
        CscMatrix {
            m,
            n,
            colptr,
            rowval,
            nzval,
        }
    }

    /// allocate space for a sparse matrix with `nnz` elements

    pub fn spalloc(size: (usize, usize), nnz: usize) -> Self {
        let (m, n) = size;
        let mut colptr = vec![0; n + 1];
        let rowval = vec![0; nnz];
        let nzval = vec![T::zero(); nnz];
        colptr[n] = nnz;

        CscMatrix::new(m, n, colptr, rowval, nzval)
    }

    /// Sparse matrix of zeros of size `m` x `n`
    pub fn zeros(size: (usize, usize)) -> Self {
        Self::spalloc(size, 0)
    }

    /// Identity matrix of size `n`
    pub fn identity(n: usize) -> Self {
        let colptr = (0usize..=n).collect();
        let rowval = (0usize..n).collect();
        let nzval = vec![T::one(); n];

        CscMatrix::new(n, n, colptr, rowval, nzval)
    }

    /// number of nonzeros
    pub fn nnz(&self) -> usize {
        self.colptr[self.n]
    }

    /// transpose
    pub fn t(&self) -> Adjoint<'_, Self> {
        Adjoint { src: self }
    }

    /// symmetric view
    pub fn sym(&self) -> Symmetric<'_, Self> {
        debug_assert!(self.is_triu());
        Symmetric { src: self }
    }

    /// Check that matrix data is correctly formatted.
    pub fn check_format(&self) -> Result<(), SparseFormatError> {
        if self.rowval.len() != self.nzval.len() {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        if self.colptr.is_empty()
            || (self.colptr.len() - 1) != self.n
            || self.colptr[self.n] != self.rowval.len()
        {
            return Err(SparseFormatError::IncompatibleDimension);
        }

        //check for colptr monotonicity
        if self.colptr.windows(2).any(|c| c[0] > c[1]) {
            return Err(SparseFormatError::BadColptr);
        }

        //check for rowval monotonicity within each column
        for col in 0..self.n {
            let rng = self.colptr[col]..self.colptr[col + 1];
            if self.rowval[rng].windows(2).any(|c| c[0] >= c[1]) {
                return Err(SparseFormatError::BadRowval);
            }
        }
        //check for row values out of bounds
        if !self.rowval.iter().all(|r| r < &self.m) {
            return Err(SparseFormatError::BadRowval);
        }

        Ok(())
    }

    /// Select a subset of the rows of a sparse matrix
    ///
    /// # Panics
    /// Panics if row dimensions are incompatible
    pub fn select_rows(&self, rowidx: &Vec<bool>) -> Self {
        //first check for compatible row dimensions
        assert_eq!(rowidx.len(), self.m);

        //count the number of rows in the reduced matrix and build an
        //index from the logical rowidx to the reduced row number
        let mut rridx = vec![0; self.m];
        let mut mred = 0;
        for (r, is_used) in zip(&mut rridx, rowidx) {
            if *is_used {
                *r = mred;
                mred += 1;
            }
        }

        // count the nonzeros in Ared
        let nzred = self.rowval.iter().filter(|&r| rowidx[*r]).count();

        // Allocate a reduced size A
        let mut Ared = CscMatrix::spalloc((mred, self.n), nzred);

        //populate new matrix
        let mut ptrred = 0;
        for col in 0..self.n {
            Ared.colptr[col] = ptrred;
            for ptr in self.colptr[col]..self.colptr[col + 1] {
                let thisrow = self.rowval[ptr];
                if rowidx[thisrow] {
                    Ared.rowval[ptrred] = rridx[thisrow];
                    Ared.nzval[ptrred] = self.nzval[ptr];
                    ptrred += 1;
                }
            }
            Ared.colptr[Ared.n] = ptrred;
        }

        Ared
    }

    /// Allocates a new matrix containing only entries from the upper triangular part
    pub fn to_triu(&self) -> Self {
        assert_eq!(self.m, self.n);
        let (m, n) = (self.m, self.n);
        let mut colptr = vec![0; n + 1];
        let mut nnz = 0;

        //count the number of entries in the upper triangle
        //and place the totals into colptr

        for col in 0..n {
            //start / stop indices for the current column
            let first = self.colptr[col];
            let last = self.colptr[col + 1];
            let rows = &self.rowval[first..last];

            // number of entries on or above diagonal in this column,
            // shifted by 1 (i.e. colptr keeps a 0 in the first column)
            colptr[col + 1] = rows.iter().filter(|&row| *row <= col).count();
            nnz += colptr[col + 1];
        }

        //allocate and copy the upper triangle entries of
        //each column into the new value vector.
        //NB! : assumes that entries in each column have
        //monotonically increasing row numbers
        let mut rowval = vec![0; nnz];
        let mut nzval = vec![T::zero(); nnz];

        for col in 0..n {
            let ntriu = colptr[col + 1];

            //start / stop indices for the destination
            let fdest = colptr[col];
            let ldest = fdest + ntriu;

            //start / stop indices for the source
            let fsrc = self.colptr[col];
            let lsrc = fsrc + ntriu;

            //copy upper triangle values
            rowval[fdest..ldest].copy_from_slice(&self.rowval[fsrc..lsrc]);
            nzval[fdest..ldest].copy_from_slice(&self.nzval[fsrc..lsrc]);

            //this should now be cumsum of the counts
            colptr[col + 1] = ldest;
        }
        CscMatrix::new(m, n, colptr, rowval, nzval)
    }

    /// True if the matrix is upper triangular
    pub fn is_triu(&self) -> bool {
        // check lower triangle for any structural entries, regardless
        // of the values that may be assigned to them
        for col in 0..self.ncols() {
            //start / stop indices for the current column
            let first = self.colptr[col];
            let last = self.colptr[col + 1];
            let rows = &self.rowval[first..last];

            // number of entries on or above diagonal in this column,
            // shifted by 1 (i.e. colptr keeps a 0 in the first column)
            if rows.iter().any(|&row| row > col) {
                return false;
            }
        }
        true
    }

    /// Returns the value at the given (row,col) index as an Option.
    /// Returns None if the given index is not a structural nonzero.
    ///
    /// # Panics
    /// Panics if the given index is out of bounds.
    pub fn get_entry(&self, idx: (usize, usize)) -> Option<T> {
        let (row, col) = idx;
        assert!(row < self.nrows() && col < self.ncols());

        let first = self.colptr[col];
        let last = self.colptr[col + 1];
        let rows_in_this_column = &self.rowval[first..last];
        match rows_in_this_column.binary_search(&row) {
            Ok(idx) => Some(self.nzval[first + idx]),
            Err(_) => None,
        }
    }
}

impl<T> ShapedMatrix for CscMatrix<T> {
    fn nrows(&self) -> usize {
        self.m
    }
    fn ncols(&self) -> usize {
        self.n
    }
    fn size(&self) -> (usize, usize) {
        (self.m, self.n)
    }
    fn shape(&self) -> MatrixShape {
        MatrixShape::N
    }
    fn is_square(&self) -> bool {
        self.m == self.n
    }
}

/// Make a concrete [CscMatrix] from its [Adjoint].   This operation will
/// allocate a new matrix and copy the data from the adjoint.
///
/// __Example usage__ : To construct the transpose of a 3 x 3 matrix:
/// ```text
/// A = [1.,  0.,  0.]
///     [2.,  4.,  0.]
///     [3.,  5.,  6.]
///```
/// ```no_run
/// use clarabel::algebra::CscMatrix;
///
/// let A : CscMatrix = (&[
///     [1., 0., 0.], //
///     [2., 4., 0.], //
///     [3., 5., 6.],
/// ]).into();
///
/// let At = A.t();  //Adjoint form.   Does not copy anything.
///
/// let B : CscMatrix = At.into(); //Concrete form.  Allocates and copies.
///
/// assert_eq!(A, B);
///
/// ```
impl<'a, T> From<Adjoint<'a, CscMatrix<T>>> for CscMatrix<T>
where
    T: FloatT,
{
    fn from(M: Adjoint<'a, CscMatrix<T>>) -> CscMatrix<T> {
        let src = M.src;

        let (m, n) = (src.n, src.m);
        let mut A = CscMatrix::spalloc((m, n), src.nnz());

        //make dummy mapping indices since we don't care
        //where the entries go
        let mut amap = vec![0usize; src.nnz()];

        A.colcount_block(src, 0, MatrixShape::T);
        A.colcount_to_colptr();
        A.fill_block(src, &mut amap, 0, 0, MatrixShape::T);
        A.backshift_colptrs();
        A
    }
}

#[test]
fn test_csc_from_slice_of_arrays() {
    let A = CscMatrix::new(
        3,                    // m
        2,                    // n
        vec![0, 2, 4],        // colptr
        vec![0, 1, 0, 2],     // rowval
        vec![1., 3., 2., 4.], // nzval
    );

    let B = CscMatrix::from(&[
        [1., 2.], //
        [3., 0.], //
        [0., 4.],
    ]); //

    let C: CscMatrix = (&[
        [1., 2.], //
        [3., 0.], //
        [0., 4.],
    ])
        .into();

    assert_eq!(A, B);
    assert_eq!(A, C);
}

#[test]
fn test_csc_get_entry() {
    let A = CscMatrix::from(&[
        [0.0, 4.0, 0.0, 0.0, 12.0],
        [1.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 6.0, 0.0, 0.0, 13.0],
        [2.0, 7.0, 10.0, 0.0, 0.0],
        [0.0, 8.0, 11.0, 0.0, 14.0],
        [3.0, 9.0, 0.0, 0.0, 0.0],
    ]);

    assert_eq!(A.get_entry((1, 0)), Some(1.));
    assert_eq!(A.get_entry((5, 0)), Some(3.));
    assert_eq!(A.get_entry((0, 1)), Some(4.));
    assert_eq!(A.get_entry((3, 1)), Some(7.));
    assert_eq!(A.get_entry((5, 1)), Some(9.));
    assert_eq!(A.get_entry((3, 2)), Some(10.));
    assert_eq!(A.get_entry((4, 2)), Some(11.));
    assert_eq!(A.get_entry((4, 4)), Some(14.));

    assert_eq!(A.get_entry((0, 0)), None);
    assert_eq!(A.get_entry((4, 0)), None);
    assert_eq!(A.get_entry((2, 2)), None);
    assert_eq!(A.get_entry((1, 3)), None);
    assert_eq!(A.get_entry((2, 3)), None);
    assert_eq!(A.get_entry((4, 3)), None);
    assert_eq!(A.get_entry((3, 4)), None);
}

#[test]
fn test_adjoint_into() {
    let A: CscMatrix = (&[
        [1., 0., 0.], //
        [2., 4., 0.], //
        [3., 5., 6.],
    ])
        .into();

    let T: CscMatrix = (&[
        [1., 2., 3.], //
        [0., 4., 5.], //
        [0., 0., 6.],
    ])
        .into();

    let B: CscMatrix = A.t().into(); //Concrete form.  Allocates and copies.

    assert_eq!(B, T);
}
