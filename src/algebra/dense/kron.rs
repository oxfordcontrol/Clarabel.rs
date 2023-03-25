#![allow(non_snake_case)]
use crate::algebra::{DenseMatrix, FloatT, Matrix, ShapedMatrix};

impl<T> Matrix<T>
where
    T: FloatT,
{
    pub fn kron<MATA, MATB>(&mut self, A: &MATA, B: &MATB) -> &Self
    where
        MATB: DenseMatrix<T = T, Output = T>,
        MATA: DenseMatrix<T = T, Output = T>,
    {
        let (pp, qq) = A.size();
        let (rr, ss) = B.size();
        assert!(self.nrows() == pp * rr);
        assert!(self.ncols() == qq * ss);

        let mut i = 0;
        for q in 0..qq {
            for s in 0..ss {
                for p in 0..pp {
                    let Apq = A[(p, q)];
                    for r in 0..rr {
                        self.data_mut()[i] = Apq * B[(r, s)];
                        i += 1;
                    }
                }
            }
        }
        self
    }
}

#[test]
fn test_kron() {
    let A = Matrix::new_from_slice((2, 2), &[1.0, 4.0, 2.0, 5.0]);
    let B = Matrix::new_from_slice((1, 2), &[1.0, 2.0]);

    let (k1, m1) = A.size();
    let (k2, m2) = B.size();

    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B);
    assert!(K.data() == vec![1., 4., 2., 8., 2., 5., 4., 10.]);
    K.kron(&A.t(), &B);
    assert!(K.data() == vec![1., 2., 2., 4., 4., 5., 8., 10.]);

    let (k2, m2) = B.t().size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B.t());
    assert!(K.data() == vec![1., 2., 4., 8., 2., 4., 5., 10.]);
    K.kron(&A.t(), &B.t());
    assert!(K.data() == vec![1., 2., 2., 4., 4., 8., 5., 10.]);
}
