use crate::algebra::{BlockConcatenate, CscMatrix, FloatT, MatrixShape};

impl<T> BlockConcatenate for CscMatrix<T>
where
    T: FloatT,
{
    fn hcat(A: &Self, B: &Self) -> Self {
        //first check for compatible row dimensions
        assert_eq!(A.m, B.m);

        //dimensions for C = [A B];
        let nnz = A.nnz() + B.nnz();
        let m = A.m; //rows C
        let n = A.n + B.n; //cols C
        let mut C = CscMatrix::spalloc((m, n), nnz);

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

    fn vcat(A: &Self, B: &Self) -> Self {
        //first check for compatible column dimensions
        assert_eq!(A.n, B.n);

        //dimensions for C = [A; B];
        let nnz = A.nnz() + B.nnz();
        let m = A.m + B.m; //rows C
        let n = A.n; //cols C
        let mut C = CscMatrix::spalloc((m, n), nnz);

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
}
