#![allow(non_snake_case)]
use super::{CscMatrix, FloatT, MatrixShape, MatrixTriangle};

impl<T> CscMatrix<T>
where
    T: FloatT,
{
    /// CscMatrix constructor.
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

    /// number of rows
    pub fn nrows(&self) -> usize {
        self.m
    }
    /// number of columns
    pub fn ncols(&self) -> usize {
        self.n
    }
    /// number of nonzeros
    pub fn nnz(&self) -> usize {
        self.colptr[self.n]
    }
    /// true if `self.nrows() == self.ncols()`
    pub fn is_square(&self) -> bool {
        self.m == self.n
    }

    /// allocate space for a sparse matrix with `nnz` elements
    ///
    /// To make an m x n matrix of zeros, use
    /// ```
    /// # use clarabel::algebra::CscMatrix;
    /// # let m = 3;
    /// # let n = 4;
    /// let A : CscMatrix<f64> = CscMatrix::spalloc(m,n,0);
    /// ```

    pub fn spalloc(m: usize, n: usize, nnz: usize) -> Self {
        let mut colptr = vec![0; n + 1];
        let rowval = vec![0; nnz];
        let nzval = vec![T::zero(); nnz];
        colptr[n] = nnz;

        CscMatrix::new(m, n, colptr, rowval, nzval)
    }

    /// Identity matrix of size `n`
    pub fn identity(n: usize) -> Self {
        let colptr = (0usize..=n).collect();
        let rowval = (0usize..n).collect();
        let nzval = vec![T::one(); n];

        CscMatrix::new(n, n, colptr, rowval, nzval)
    }

    /// horizontal matrix concatenation:
    /// ```text
    /// C = [A B]
    /// ```
    /// # Panics
    /// Panics if row dimensions are incompatible

    pub fn hcat(A: &Self, B: &Self) -> Self {
        //first check for compatible row dimensions
        assert_eq!(A.m, B.m);

        //dimensions for C = [A B];
        let nnz = A.nnz() + B.nnz();
        let m = A.m; //rows
        let n = A.n + B.n; //cols s
        let mut C = CscMatrix::spalloc(m, n, nnz);

        //we make dummy mapping indices since we don't care
        //where the entries go.  An alternative would be to
        //modify the fill_block method to accept Option<_>
        let mut amap = vec![0usize; A.nnz()];
        let mut bmap = vec![0usize; B.nnz()];

        //compute column counts and fill
        C.colcount_block(A, 0, MatrixShape::N);
        C.colcount_block(B, A.n, MatrixShape::N);
        C.colcount_to_colptr();

        C.fill_block(A, &mut amap, 0, 0, MatrixShape::N);
        C.fill_block(B, &mut bmap, 0, A.n, MatrixShape::N);
        C.backshift_colptrs();

        C
    }

    /// vertical matrix concatenation:
    /// ```text
    /// C = [ A ]
    ///     [ B ]
    /// ```
    ///
    /// # Panics
    /// Panics if column dimensions are incompatible

    pub fn vcat(A: &Self, B: &Self) -> Self {
        //first check for compatible column dimensions
        assert_eq!(A.n, B.n);

        //dimensions for C = [A; B];
        let nnz = A.nnz() + B.nnz();
        let m = A.m + B.m; //rows
        let n = A.n; //cols s
        let mut C = CscMatrix::spalloc(m, n, nnz);

        //we make dummy mapping indices since we don't care
        //where the entries go.  An alternative would be to
        //modify the fill_block method to accept Option<_>
        let mut amap = vec![0usize; A.nnz()];
        let mut bmap = vec![0usize; B.nnz()];

        //compute column counts and fill
        C.colcount_block(A, 0, MatrixShape::N);
        C.colcount_block(B, 0, MatrixShape::N);
        C.colcount_to_colptr();

        C.fill_block(A, &mut amap, 0, 0, MatrixShape::N);
        C.fill_block(B, &mut bmap, A.m, 0, MatrixShape::N);
        C.backshift_colptrs();
        C
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

        for col in 0..self.n {
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
}

//---------------------------------------------------------
// The remainder of the CscMatrix implementation consists of
// low-level utilities for counting / filling entries in
// block partitioned sparse matrices.  NB:  this is still
// impl<T> CscMatrix<T>, as above, but everything is
// pub(crate)
//---------------------------------------------------------

impl<T> CscMatrix<T>
where
    T: FloatT,
{
    // increment self.colptr by the number of nonzeros
    // in a dense upper/lower triangle on the diagonal.
    pub(crate) fn colcount_dense_triangle(
        &mut self,
        initcol: usize,
        blockcols: usize,
        shape: MatrixTriangle,
    ) {
        let cols = self.colptr[(initcol)..(initcol + blockcols)].iter_mut();
        let counts = 1..(blockcols + 1);
        match shape {
            MatrixTriangle::Triu => {
                cols.zip(counts).for_each(|(x, c)| *x += c);
            }

            MatrixTriangle::Tril => {
                cols.zip(counts.rev()).for_each(|(x, c)| *x += c);
            }
        }
    }

    // increment the self.colptr by the number of nonzeros
    // in a square diagonal matrix placed on the diagonal.
    pub(crate) fn colcount_diag(&mut self, initcol: usize, blockcols: usize) {
        let cols = self.colptr[initcol..(initcol + blockcols)].iter_mut();
        cols.for_each(|x| *x += 1);
    }

    // same as kkt_count_diag, but counts places
    // where the input matrix M has a missing
    // diagonal entry.  M must be square and TRIU
    pub(crate) fn colcount_missing_diag(&mut self, M: &CscMatrix<T>, initcol: usize) {
        assert_eq!(M.colptr.len(), M.n + 1);
        assert!(self.colptr.len() >= M.n + initcol);

        for i in 0..M.n {
            if M.colptr[i] == M.colptr[i+1] ||    // completely empty column
               M.rowval[M.colptr[i+1]-1] != i
            // last element is not on diagonal
            {
                self.colptr[i + initcol] += 1;
            }
        }
    }

    // increment the self.colptr by the a number of nonzeros.
    // used to account for the placement of a column
    // vector that partially populates the column
    pub(crate) fn colcount_colvec(&mut self, n: usize, _firstrow: usize, firstcol: usize) {
        // just add the vector length to this column
        self.colptr[firstcol] += n;
    }

    // increment the self.colptr by 1 for every element
    // used to account for the placement of a column
    // vector that partially populates the column
    pub(crate) fn colcount_rowvec(&mut self, n: usize, _firstrow: usize, firstcol: usize) {
        // add one element to each of n consective columns
        // starting from initcol.  The row index doesn't
        // matter here.
        let cols = self.colptr[firstcol..(firstcol + n)].iter_mut();
        cols.for_each(|x| *x += 1);
    }

    // increment the self.colptr by the number of nonzeros in M

    pub(crate) fn colcount_block(&mut self, M: &CscMatrix<T>, initcol: usize, shape: MatrixShape) {
        match shape {
            MatrixShape::T => {
                for row in M.rowval.iter() {
                    self.colptr[initcol + row] += 1;
                }
            }
            MatrixShape::N => {
                // just add the column count
                for i in 0..M.n {
                    self.colptr[initcol + i] += M.colptr[i + 1] - M.colptr[i];
                }
            }
        }
    }

    //populate a partial column with zeros using the self.colptr as the indicator
    // the next fill location in each row.
    pub(crate) fn fill_colvec(&mut self, vtoKKT: &mut [usize], initrow: usize, initcol: usize) {
        for (i, v) in vtoKKT.iter_mut().enumerate() {
            let dest = self.colptr[initcol];
            self.rowval[dest] = initrow + i;
            self.nzval[dest] = T::zero();
            *v = dest;
            self.colptr[initcol] += 1;
        }
    }

    // populate a partial row with zeros using the self.colptr as indicator of
    // next fill location in each row.
    pub(crate) fn fill_rowvec(&mut self, vtoKKT: &mut [usize], initrow: usize, initcol: usize) {
        for (i, v) in vtoKKT.iter_mut().enumerate() {
            let col = initcol + i;
            let dest = self.colptr[col];
            self.rowval[dest] = initrow;
            self.nzval[dest] = T::zero();
            *v = dest;
            self.colptr[col] += 1;
        }
    }

    // populate values from M using the self.colptr as indicator of
    // next fill location in each row.
    pub(crate) fn fill_block(
        &mut self,
        M: &CscMatrix<T>,
        MtoKKT: &mut [usize],
        initrow: usize,
        initcol: usize,
        shape: MatrixShape,
    ) {
        for i in 0..M.n {
            let z = M.rowval.iter().zip(M.nzval.iter());
            let start = M.colptr[i];
            let stop = M.colptr[i + 1];

            for (j, (&Mrow, &Mval)) in z.take(stop).skip(start).enumerate() {
                let (col, row);

                match shape {
                    MatrixShape::T => {
                        col = Mrow + initcol;
                        row = i + initrow;
                    }
                    MatrixShape::N => {
                        col = i + initcol;
                        row = Mrow + initrow;
                    }
                };

                let dest = self.colptr[col];
                self.rowval[dest] = row;
                self.nzval[dest] = Mval;
                self.colptr[col] += 1;
                MtoKKT[j] = dest;
            }
        }
    }

    // Populate the upper or lower triangle with 0s using the self.colptr
    // as indicator of next fill location in each row
    pub(crate) fn fill_dense_triangle(
        &mut self,
        blocktoKKT: &mut [usize],
        offset: usize,
        blockdim: usize,
        shape: MatrixTriangle,
    ) {
        // data will always be supplied as triu, so when filling it into
        // a tril shape we also need to transpose it.   Just write two
        // separate functions for clarity here

        match shape {
            MatrixTriangle::Triu => {
                self._fill_dense_triangle_triu(blocktoKKT, offset, blockdim);
            }

            MatrixTriangle::Tril => {
                self._fill_dense_triangle_tril(blocktoKKT, offset, blockdim);
            }
        }
    }

    pub(crate) fn _fill_dense_triangle_triu(
        &mut self,
        blocktoKKT: &mut [usize],
        offset: usize,
        blockdim: usize,
    ) {
        let mut kidx = 0;
        for col in offset..(offset + blockdim) {
            for row in offset..=col {
                let dest = self.colptr[col];
                self.rowval[dest] = row;
                self.nzval[dest] = T::zero(); //structural zero
                self.colptr[col] += 1;
                blocktoKKT[kidx] = dest;
                kidx += 1;
            }
        }
    }

    pub(crate) fn _fill_dense_triangle_tril(
        &mut self,
        blocktoKKT: &mut [usize],
        offset: usize,
        blockdim: usize,
    ) {
        let mut kidx = 0;
        for col in offset..(offset + blockdim) {
            for row in offset..=col {
                let dest = self.colptr[col];
                self.rowval[dest] = row;
                self.nzval[dest] = T::zero(); //structural zero
                self.colptr[col] += 1;
                blocktoKKT[kidx] = dest;
                kidx += 1;
            }
        }
    }

    // Populate the diagonal with 0s using the K.colptr as indicator of
    // next fill location in each row
    pub(crate) fn fill_diag(&mut self, diagtoKKT: &mut [usize], offset: usize, blockdim: usize) {
        for (i, col) in (offset..(offset + blockdim)).enumerate() {
            let dest = self.colptr[col];
            self.rowval[dest] = col;
            self.nzval[dest] = T::zero(); //structural zero
            self.colptr[col] += 1;
            diagtoKKT[i] = dest;
        }
    }

    // same as fill_diag, but only places zero
    // entries where the input matrix M has a missing
    // diagonal entry.  M must be square and TRIU
    pub(crate) fn fill_missing_diag(&mut self, M: &CscMatrix<T>, initcol: usize) {
        for i in 0..M.n {
            // fill out missing diagonal terms only
            if M.colptr[i] == M.colptr[i+1] ||    // completely empty column
               M.rowval[M.colptr[i+1]-1] != i
            {
                // last element is not on diagonal
                let dest = self.colptr[i + initcol];
                self.rowval[dest] = i + initcol;
                self.nzval[dest] = T::zero(); //structural zero
                self.colptr[i] += 1;
            }
        }
    }

    pub(crate) fn colcount_to_colptr(&mut self) {
        let mut currentptr = 0;
        for p in &mut self.colptr {
            let count = *p;
            *p = currentptr;
            currentptr += count;
        }
    }

    pub(crate) fn backshift_colptrs(&mut self) {
        self.colptr.rotate_right(1);
        self.colptr[0] = 0;
    }

    pub(crate) fn count_diagonal_entries(&self) -> usize {
        let mut count = 0;
        for i in 0..self.n {
            // compare last entry in each column with
            // its row number to identify diagonal entries
            if self.colptr[i+1] != self.colptr[i] &&    // nonempty column
               self.rowval[self.colptr[i+1]-1] == i
            {
                // last element is on diagonal
                count += 1;
            }
        }
        count
    }
}
