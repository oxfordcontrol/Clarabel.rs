#![allow(non_snake_case)]
use crate::algebra::{BlockConcatenate, FloatT, Matrix, ShapedMatrix};
use std::iter::Extend;

impl<T> BlockConcatenate for Matrix<T>
where
    T: FloatT,
{
    fn hcat(A: &Self, B: &Self) -> Self {
        //first check for compatible row dimensions
        assert_eq!(A.m, B.m);

        //dimensions for C = [A B];
        let m = A.m; //rows
        let n = A.n + B.n; //cols s
        let mut data = A.data.clone();
        data.extend(&B.data);
        Self { m, n, data }
    }

    fn vcat(A: &Self, B: &Self) -> Self {
        //first check for compatible column dimensions
        assert_eq!(A.n, B.n);

        //dimensions for C = [A; B];
        let m = A.m + B.m; //rows C
        let n = A.n; //cols C
        let mut data = vec![];
        data.reserve_exact(m * n);

        for col in 0..A.ncols() {
            data.extend(A.col_slice(col));
            data.extend(B.col_slice(col));
        }
        Self { m, n, data }
    }
}

#[test]
fn test_dense_concatenate() {
    let A = Matrix::from(&[
        [1., 3.], //
        [2., 4.], //
    ]);
    let B = Matrix::from(&[
        [5., 7.], //
        [6., 8.], //
    ]);

    let C = Matrix::hcat(&A, &B);

    let Ctest = Matrix::from(&[
        [1., 3., 5., 7.], //
        [2., 4., 6., 8.], //
    ]);

    assert_eq!(C, Ctest);

    let C = Matrix::vcat(&A, &B);

    let Ctest = Matrix::from(&[
        [1., 3.], //
        [2., 4.], //
        [5., 7.], //
        [6., 8.], //
    ]);

    assert_eq!(C, Ctest);
}
