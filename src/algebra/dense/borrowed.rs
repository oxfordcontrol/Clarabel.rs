use crate::algebra::*;

impl<'a, T> BorrowedMatrix<'a, T>
where
    T: FloatT,
{
    pub fn from_slice(data: &'a [T], m: usize, n: usize) -> Self {
        Self {
            size: (m, n),
            data,
            phantom: std::marker::PhantomData::<T>,
        }
    }
}

impl<'a, T> BorrowedMatrixMut<'a, T>
where
    T: FloatT,
{
    pub fn from_slice_mut(data: &'a mut [T], m: usize, n: usize) -> Self {
        Self {
            size: (m, n),
            data,
            phantom: std::marker::PhantomData::<T>,
        }
    }
}
