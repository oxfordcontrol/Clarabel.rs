#![allow(non_snake_case)]
use crate::algebra::{DenseMatrix, FloatT, Matrix, ShapedMatrix};

impl<T> Matrix<T>
where
    T: FloatT,
{
    pub(crate) fn kron<MATA, MATB>(&mut self, A: &MATA, B: &MATB) -> &Self
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
#[rustfmt::skip]
fn test_kron() {

    let A = Matrix::from(
        &[[ 1.,  2.],
          [ 4.,  5.]]);

    let B = Matrix::from(
        &[[ 1.,  2.]]);


    // A ⊗ B
    let (k1, m1) = A.size();
    let (k2, m2) = B.size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B);

    let Ktest = Matrix::from(
        &[[ 1.,  2.,  2.,  4.],
          [ 4.,  8.,  5., 10.]]);

    assert_eq!(K,Ktest);

    // A' ⊗ B
    let (k1, m1) = A.t().size();
    let (k2, m2) = B.size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A.t(), &B);

    let Ktest = Matrix::from(
        &[[ 1.,  2.,  4.,  8.],
          [ 2.,  4.,  5., 10.]]);
          
    assert_eq!(K,Ktest);

    // A ⊗ B'            
    let (k1, m1) = A.size();
    let (k2, m2) = B.t().size();
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));
    K.kron(&A, &B.t());

    let Ktest = Matrix::from(
        &[[1., 2. ],
          [2., 4. ],
          [4., 5. ],
          [8., 10.]]);
          
    assert_eq!(K,Ktest);

    // A' ⊗ B'  
    let (k1, m1) = A.t().size();
    let (k2, m2) = B.t().size();  
    let mut K = Matrix::<f64>::zeros((k1 * k2, m1 * m2));  
    K.kron(&A.t(), &B.t());

    let Ktest = Matrix::from(
        &[[1., 4. ],
          [2., 8. ],
          [2., 5. ],
          [4., 10.]]);

    assert_eq!(K,Ktest);
}
