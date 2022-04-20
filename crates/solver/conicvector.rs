use crate::algebra::*;
use crate::cones::*;
use std::ops::{Deref, DerefMut};

// -------------------------------------
// vectors defined w.r.t. to conic constraints
// includes start/stop indices of the subvector
// ---------------------------------------

pub struct ConicVector<T: FloatT = f64> {
    //contiguous array of source data
    vec: Vec<T>,

    //array of start/stop tuples of the subvectors
    //NB: this different from the Julia prototype,
    //which instead used a vector of views.  Rust
    //borrow checker does not want to allow simultaneous
    //mutable access to multiple views of the data
    ranges: Vec<(usize, usize)>,
}

impl<T: FloatT> ConicVector<T> {
    pub fn new(set: &ConeSet<T>) -> Self {
        // make an internal copy to protect from user modification
        let vec = vec![T::zero(); set.numel()];
        let mut ranges = Vec::with_capacity(set.len());

        //NB: last index is one past the final element
        //required for each range.   This differs from
        //the Julia implementation
        let mut first = 0;
        for cone in set.iter() {
            let last = first + cone.numel();
            ranges.push((first, last));
            first = last;
        }

        Self {
            vec: vec,
            ranges: ranges,
        }
    }

    pub fn view(&self, i: usize) -> &[T] {
        &self.vec[self.ranges[i].0..self.ranges[i].1]
    }

    pub fn view_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.vec[self.ranges[i].0..self.ranges[i].1]
    }
}

impl<T: FloatT> Deref for ConicVector<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.vec
    }
}

impl<T: FloatT> DerefMut for ConicVector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vec
    }
}
