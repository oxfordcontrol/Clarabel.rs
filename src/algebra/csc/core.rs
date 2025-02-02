#![allow(non_snake_case)]

use crate::algebra::permute;
use crate::algebra::utils::sortperm_by;
use crate::algebra::{Adjoint, MatrixShape, ShapedMatrix, SparseFormatError, Symmetric};
use num_traits::Num;
use std::iter::{repeat, zip};

#[cfg(feature = "serde")]
use serde::{de::DeserializeOwned, Deserialize, Serialize};

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

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = "T: Serialize + DeserializeOwned"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscMatrix<T = f64> {
    /// number of rows
    pub m: usize,
    /// number of columns
    pub n: usize,
    /// CSC format column pointer.   
    ///
    /// This field should have length `n+1`. The last entry corresponds
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
    T: Num + Copy + 'a,
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
        let nnz = rows.iter().flatten().filter(|&v| *v != T::zero()).count();

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
    T: Num + Copy,
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

    /// `CscMatrix` constructor from data in triplet format.
    ///
    /// # Panics
    /// Makes rudimentary dimensional compatibility checks and panics on
    /// failure.   Data can be provided unsorted.   Repeated values are added.
    ///
    pub fn new_from_triplets(m: usize, n: usize, I: Vec<usize>, J: Vec<usize>, V: Vec<T>) -> Self {
        assert_eq!(I.len(), J.len());
        assert_eq!(I.len(), V.len());

        let mut M = CscMatrix::spalloc((m, n), V.len());

        let mut p = vec![0; V.len()];

        // use M.rowptr as temporary workspace
        M.rowval.iter_mut().enumerate().for_each(|(i, p)| *p = i);

        // sort by column, then by row
        sortperm_by(&mut p, &M.rowval, |&a, &b| {
            J[a].cmp(&J[b]).then(I[a].cmp(&I[b]))
        });

        // map data into the matrix in sorted order
        permute(&mut M.rowval, &I, &p);
        permute(&mut M.nzval, &V, &p);

        // assemble the column counts
        for &c in J.iter() {
            M.colptr[c] += 1;
        }

        // make a second pass to consolidate repeated entries
        // within each column
        let mut readidx = 0;
        let mut writeidx = 0;

        for col in 0..n {
            let nentries = M.colptr[col]; //entries in this column
            for j in 0..nentries {
                // non-repeated or first entry in column
                if j == 0 || M.rowval[readidx] != M.rowval[readidx - 1] {
                    if writeidx != readidx {
                        M.rowval[writeidx] = M.rowval[readidx];
                        M.nzval[writeidx] = M.nzval[readidx];
                    }
                    writeidx += 1;
                    readidx += 1;
                }
                // repeated row entry with value to be consolidated
                else {
                    M.nzval[writeidx - 1] = M.nzval[writeidx - 1] + M.nzval[readidx];
                    M.colptr[col] -= 1;
                    readidx += 1;
                }
            }
        }

        M.rowval.resize(writeidx, 0);
        M.nzval.resize(writeidx, T::zero());

        M.colcount_to_colptr();

        M
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

    /// squeeze out entries that are == T::zero()
    pub fn dropzeros(&mut self) {
        // this function could possibly be generalized to allow filtering
        // on a more general test, similar to fkeep! in Julia sparse matrix
        // internals.  Then could be used as a filter for triu matrix etc

        // Sweep through columns, rewriting kept elements in their new positions
        // and updating the column pointers accordingly as we go.
        let mut writeidx: usize = 0;
        let mut first: usize = 0;

        for col in 0..self.ncols() {
            let last = self.colptr[col + 1];

            for readidx in first..last {
                let val = self.nzval[readidx];
                let row = self.rowval[readidx];

                // If nonzero and a shift so far, move the value
                if val != T::zero() {
                    if writeidx != readidx {
                        self.nzval[writeidx] = val;
                        self.rowval[writeidx] = row;
                    }
                    writeidx += 1;
                }
            }

            first = self.colptr[col + 1];
            self.colptr[col + 1] = writeidx;
        }

        self.rowval.resize(writeidx, 0);
        self.nzval.resize(writeidx, T::zero());
    }

    /// Return matrix data in triplet format.
    ///
    #[cfg_attr(not(feature = "sdp"), allow(dead_code))]
    pub(crate) fn findnz(&self) -> (Vec<usize>, Vec<usize>, Vec<T>) {
        let I = self.rowval.clone();
        let mut J = Vec::with_capacity(self.nnz());
        let V = self.nzval.clone();

        for c in 0..self.ncols() {
            let times = self.colptr[c + 1] - self.colptr[c];
            J.extend(repeat(c).take(times));
        }
        (I, J, V)
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

    /// Check that matrix data is canonically formatted.
    pub fn check_format(&self) -> Result<(), SparseFormatError> {
        self.check_dimensions()?;

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

    /// Put matrix into standard ('canonical') form, operating in place.  This function
    /// sorts data within each column by row index, and removes any duplicates.
    /// Does not remove structural zeros.
    ///
    /// # Panics
    /// Panics if the matrix initial dimensions are incompatible.
    ///
    pub fn canonicalize(&mut self) -> Result<(), SparseFormatError> {
        self.check_dimensions()?;
        self.sort_indices()?;
        self.deduplicate()
    }

    /// Adds together repeated entries in the same column.   Input must
    /// already be in column sorted order.
    fn sort_indices(&mut self) -> Result<(), SparseFormatError> {
        let mut tempdata: Vec<(usize, T)> = Vec::new();

        for col in 0..self.n {
            let start = self.colptr[col];
            let stop = self.colptr[col + 1];

            let nzval = &mut self.nzval[start..stop];
            let rowval = &mut self.rowval[start..stop];

            tempdata.resize(stop - start, (0, T::zero()));

            for (i, (r, v)) in zip(rowval.iter(), nzval.iter()).enumerate() {
                tempdata[i] = (*r, *v);
            }
            tempdata.sort_by_key(|&(r, _)| r);

            for (i, (r, v)) in tempdata.iter().enumerate() {
                rowval[i] = *r;
                nzval[i] = *v;
            }
        }

        Ok(())
    }

    /// Adds together repeated entries in the same column.   Input must
    /// already be in column sorted order.
    fn deduplicate(&mut self) -> Result<(), SparseFormatError> {
        let mut nnz = 0;
        let mut stop = 0;

        for col in 0..self.n {
            let mut ptr = stop;
            stop = self.colptr[col + 1];

            while ptr < stop {
                let thisrow = self.rowval[ptr];
                let mut accum = self.nzval[ptr];
                ptr += 1;

                while (ptr < stop) && (self.rowval[ptr] == thisrow) {
                    accum = accum + self.nzval[ptr];
                    ptr += 1;
                }
                self.rowval[nnz] = thisrow;
                self.nzval[nnz] = accum;
                nnz += 1;
            }
            self.colptr[col + 1] = nnz;
        }

        self.rowval.truncate(nnz);
        self.nzval.truncate(nnz);

        Ok(())
    }

    /// Check that for dimensional consistency.  Private since users should
    /// check everything via check_format, and the canonicalization functions
    /// must at least check dimensions before running.
    fn check_dimensions(&self) -> Result<(), SparseFormatError> {
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
        Ok(())
    }

    /// True if matrices if the same size and sparsity pattern
    pub fn is_equal_sparsity(&self, other: &Self) -> bool {
        self.size() == other.size() && self.colptr == other.colptr && self.rowval == other.rowval
    }

    /// Same as `is_equal_sparsity`, but returns an error indicating the reason
    /// for failure if the matrices do not have equivalent sparsity patterns.
    pub fn check_equal_sparsity(&self, other: &Self) -> Result<(), SparseFormatError> {
        if self.size() != other.size() {
            Err(SparseFormatError::IncompatibleDimension)
        } else if self.colptr != other.colptr || self.rowval != other.rowval {
            Err(SparseFormatError::SparsityMismatch)
        } else {
            Ok(())
        }
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

    /// Sets a value at a given (row,col) index, allocating
    /// additional space in the matrix if required.  
    ///
    /// # Panics
    /// Panics if the given index is out of bounds.
    pub fn set_entry(&mut self, idx: (usize, usize), value: T) {
        let (row, col) = idx;
        assert!(row < self.nrows() && col < self.ncols());

        let first = self.colptr[col];
        let last = self.colptr[col + 1];
        let rows_in_this_column = &self.rowval[first..last];

        let i = rows_in_this_column.partition_point(|&x| x < row);

        if i == rows_in_this_column.len() || rows_in_this_column[i] != row {
            // don't allocate space for insertion of new zeros
            if value == T::zero() {
                return;
            }

            // the element must be inserted, then col counts rebuilt
            self.rowval.insert(first + i, row);
            self.nzval.insert(first + i, value);

            // a bit wasteful since we only really need to
            // rebuil from the insertion point onwards
            self.colptr_to_colcount();
            self.colptr[col] += 1;
            self.colcount_to_colptr();
        } else {
            // the element already exists, so overwrite it
            self.nzval[first + i] = value;
        }
    }

    /// Returns the (row,col) coordinates of the given linear index.
    ///
    /// # Panics
    /// Panics if the given index is out of bounds.
    pub fn index_to_coord(&self, idx: usize) -> (usize, usize) {
        assert!(idx < self.nnz());
        let row = self.rowval[idx];
        let col = self.colptr.partition_point(|&c| idx + 1 > c) - 1;
        (row, col)
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
    T: Num + Copy,
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
fn test_csc_set_entry() {
    let mut A = CscMatrix::from(&[
        [0.0, 3.0, 6.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 7.0, 8.0],
        [2.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);

    let B = CscMatrix::from(&[
        [0.0, 3.0, -6.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 7.0, -8.0],
        [2.0, 5.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 11.0],
    ]);

    // overwrite existing entries
    A.set_entry((0, 2), -6.0);
    A.set_entry((2, 3), -8.0);

    // add new entries
    A.set_entry((3, 2), 10.0);
    A.set_entry((4, 3), 11.0);

    assert_eq!(A, B);
}

#[test]
fn test_csc_index_to_coord() {
    let A = CscMatrix::from(&[
        [0.0, 4.0, 0.0, 0.0, 12.0],
        [1.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 6.0, 0.0, 0.0, 13.0],
        [2.0, 7.0, 10.0, 0.0, 0.0],
        [0.0, 8.0, 11.0, 0.0, 14.0],
        [3.0, 9.0, 0.0, 0.0, 0.0],
    ]);

    assert_eq!(A.index_to_coord(0), (1, 0));
    assert_eq!(A.index_to_coord(1), (3, 0));
    assert_eq!(A.index_to_coord(2), (5, 0));
    assert_eq!(A.index_to_coord(3), (0, 1));
    assert_eq!(A.index_to_coord(4), (1, 1));
    assert_eq!(A.index_to_coord(5), (2, 1));
    assert_eq!(A.index_to_coord(6), (3, 1));
    assert_eq!(A.index_to_coord(7), (4, 1));
    assert_eq!(A.index_to_coord(8), (5, 1));
    assert_eq!(A.index_to_coord(9), (3, 2));
    assert_eq!(A.index_to_coord(10), (4, 2));
    assert_eq!(A.index_to_coord(11), (0, 4));
    assert_eq!(A.index_to_coord(12), (2, 4));
    assert_eq!(A.index_to_coord(13), (4, 4));
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

#[test]
fn test_triplets() {
    let A: CscMatrix = (&[
        [1., 0., 0., 5.], //
        [0., 0., 3., 0.], //
        [2., 0., 4., 0.],
    ])
        .into();

    let cols = vec![0, 0, 2, 2, 3];
    let rows = vec![0, 2, 1, 2, 0];
    let vals = vec![1., 2., 3., 4., 5.];

    // extract triplet format data and compare
    let (I, J, V) = A.findnz();
    assert_eq!(I, rows);
    assert_eq!(J, cols);
    assert_eq!(V, vals);

    // construct from triplets and compare
    let B: CscMatrix = CscMatrix::new_from_triplets(3, 4, rows, cols, vals);
    assert_eq!(A, B);

    // same thing, but with data in the wrong order
    let cols = vec![2, 0, 2, 0, 3];
    let rows = vec![2, 2, 1, 0, 0];
    let vals = vec![4., 2., 3., 1., 5.];

    let B: CscMatrix = CscMatrix::new_from_triplets(3, 4, rows, cols, vals);

    assert_eq!(A, B);

    // case with repeated entries, unsorted

    let A: CscMatrix<isize> = (&[
        [0, 0, 0],   //
        [-20, 0, 0], //
        [-20, -20, 0],
    ])
        .into();

    let rows = vec![1, 2, 2, 1, 2, 2];
    let cols = vec![0, 0, 1, 0, 0, 1];
    let vals = vec![-10, -10, -10, -10, -10, -10];

    let B = CscMatrix::new_from_triplets(3, 3, rows, cols, vals);
    assert_eq!(A, B);
}

#[test]
fn test_drop_zeros() {
    let mut A = CscMatrix::from(&[
        [0.0, 3.0, 6.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 7.0, 8.0],
        [2.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);

    // same, but with 2,6,7,8 set to zero
    let B = CscMatrix::from(&[
        [0.0, 3.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);

    // overwrite existing entries
    let dropped = [2, 6, 7, 8];
    for idx in dropped {
        A.nzval[idx - 1] = 0.0;
    }

    //squeeze out the zeros
    A.dropzeros();

    assert_eq!(A, B);
}

#[test]
fn test_sort_indices() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 2, 4, 5],
        rowval: vec![3, 1, 4, 2, 2],
        nzval: vec![2.0, 3.0, 1.0, 4.0, 5.0],
    };

    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![1, 3, 2, 4, 2]);
    assert_eq!(A.nzval, vec![3.0, 2.0, 4.0, 1.0, 5.0]);

    //nothing to sort
    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![1, 3, 2, 4, 2]);
    assert_eq!(A.nzval, vec![3.0, 2.0, 4.0, 1.0, 5.0]);
}

#[test]
fn test_sort_indices_with_duplicates() {
    let mut A = CscMatrix {
        m: 4,
        n: 2,
        colptr: vec![0, 3, 5],
        rowval: vec![3, 3, 1, 2, 4],
        nzval: vec![2.0, 3.0, 1.0, 1.0, 4.0],
    };

    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![1, 3, 3, 2, 4]);
    assert_eq!(A.nzval, vec![1.0, 2.0, 3.0, 1.0, 4.0]);
}

#[test]
fn test_deduplicate() {
    let mut A = CscMatrix {
        m: 4,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![1, 1, 2, 4],
        nzval: vec![3.0, 2.0, 1.0, 4.0],
    };

    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 1, 3]);
    assert_eq!(A.rowval, vec![1, 2, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0]);

    // nothing to deduplicate
    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 1, 3]);
    assert_eq!(A.rowval, vec![1, 2, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0]);
}

#[test]
fn test_deduplicate_multiple_columns() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 2, 4, 6],
        rowval: vec![1, 1, 2, 4, 3, 3],
        nzval: vec![3.0, 2.0, 1.0, 4.0, 5.0, 6.0],
    };

    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 1, 3, 4]);
    assert_eq!(A.rowval, vec![1, 2, 4, 3]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0, 11.0]);
}

#[test]
fn test_deduplicate_1col() {
    let mut A = CscMatrix {
        m: 4,
        n: 1,
        colptr: vec![0, 3],
        rowval: vec![1, 1, 4],
        nzval: vec![2.0, 3.0, 4.0],
    };

    A.deduplicate().unwrap();
    assert_eq!(A.colptr, vec![0, 2]);
    assert_eq!(A.rowval, vec![1, 4]);
    assert_eq!(A.nzval, vec![5.0, 4.0]);
}

#[test]
fn test_canonicalize() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 3, 4, 7],
        rowval: vec![2, 1, 1, 4, 3, 4, 3],
        nzval: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    };

    A.canonicalize().unwrap();
    assert_eq!(A.colptr, vec![0, 2, 3, 5]);
    assert_eq!(A.rowval, vec![1, 2, 4, 3, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 4.0, 12.0, 6.0]);
}

#[test]
fn test_canonicalize_structural_zeros() {
    let mut A = CscMatrix {
        m: 4,
        n: 3,
        colptr: vec![0, 3, 4, 7],
        rowval: vec![2, 1, 1, 4, 3, 4, 3],
        nzval: vec![1.0, 2.0, 3.0, 0.0, 5.0, 6.0, -5.0],
    };

    A.canonicalize().unwrap();
    assert_eq!(A.colptr, vec![0, 2, 3, 5]);
    assert_eq!(A.rowval, vec![1, 2, 4, 3, 4]);
    assert_eq!(A.nzval, vec![5.0, 1.0, 0.0, 0.0, 6.0]);
}

#[test]
fn test_canonicalize_empty() {
    let mut A: CscMatrix<f64> = CscMatrix {
        m: 0,
        n: 0,
        colptr: vec![0],
        rowval: vec![],
        nzval: vec![],
    };

    A.canonicalize().unwrap();
    assert!(A.rowval.is_empty());
    assert!(A.nzval.is_empty());
}

#[test]
fn test_canonicalize_singleton() {
    let mut A = CscMatrix {
        m: 4,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![2],
        nzval: vec![5.0],
    };

    A.sort_indices().unwrap();
    assert_eq!(A.rowval, vec![2]);
    assert_eq!(A.nzval, vec![5.0]);
}
