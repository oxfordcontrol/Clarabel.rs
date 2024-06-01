//! Internal utility functions and helpers.

// assorted other functionality missing from std

// a drop-in replacement for the julia "findall" function,
// which serves as a vectorized version of the std::iter::position
// returning indices of *all* elements satisfying a predicate

use crate::qdldl;
use num_traits::Num;
use std::cmp::Ordering;

#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) trait PositionAll<T>: Iterator<Item = T> {
    fn position_all<F>(&mut self, predicate: F) -> Vec<usize>
    where
        F: FnMut(&T) -> bool;
}

impl<T, I> PositionAll<T> for I
where
    I: Iterator<Item = T>,
{
    fn position_all<F>(&mut self, mut f: F) -> Vec<usize>
    where
        F: FnMut(&T) -> bool,
    {
        self.enumerate()
            .filter(|(_, item)| f(item))
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    }
}

// permutation and inverse permutation
pub(crate) fn permute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    qdldl::permute(x, b, p);
}

#[allow(dead_code)]
pub(crate) fn ipermute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    qdldl::ipermute(x, b, p);
}

// Construct an inverse permutation from a permutation
#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) fn invperm(p: &[usize]) -> Vec<usize> {
    let mut b = vec![0; p.len()];
    for (i, j) in p.iter().enumerate() {
        assert!(*j < p.len() && b[*j] == 0);
        b[*j] = i;
    }
    b
}

#[allow(dead_code)]
pub(crate) fn sortperm<T>(p: &mut [usize], v: &[T])
where
    T: Sized + Ord + Copy,
{
    assert_eq!(p.len(), v.len());
    p.iter_mut().enumerate().for_each(|(i, p)| *p = i);
    p.sort_by_key(|&k| v[k]);
}

#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) fn sortperm_rev<T>(p: &mut [usize], v: &[T])
where
    T: Sized + Ord + Copy,
{
    assert_eq!(p.len(), v.len());
    sortperm_by(p, v, |&a, &b| b.cmp(&a));
}

pub(crate) fn sortperm_by<T, F>(p: &mut [usize], v: &[T], compare: F)
where
    F: FnMut(&T, &T) -> Ordering,
{
    assert_eq!(p.len(), v.len());
    p.iter_mut().enumerate().for_each(|(i, p)| *p = i);
    let mut f = compare;
    p.sort_by(|&i, &j| f(&v[i], &v[j]));
}

// PJG: maybe should be a vector trait, but T needs to admit
// non-float types (e.g. usize).  Would require partition of the
// vector math traits into those that require FloatT and those
// that only require Num + Ord.
#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) fn findmax<T>(v: &[T]) -> Option<usize>
where
    T: Num + Copy + Ord,
{
    v.iter()
        .enumerate()
        .max_by_key(|(_, &value)| value)
        .map(|(idx, _)| idx)
}

// -------------
// testing

#[test]
fn test_position_all() {
    let test = [3, 1, 0, 5, 9];
    let idx = test.iter().position_all(|&v| *v > 2);
    assert_eq!(idx, vec![0, 3, 4]);

    let idx: Vec<usize> = test.iter().position_all(|&v| *v == 2);
    assert_eq!(idx, Vec::<usize>::new());
}

#[test]
fn test_permute() {
    let mut x = vec![0; 5];
    let b = [6, 7, 8, 9, 10];
    let p = [2, 4, 1, 3, 0];

    permute(&mut x, &b, &p);

    assert_eq!(x, [8, 10, 7, 9, 6]);
}

#[test]
fn test_ipermute() {
    let mut x = vec![0; 5];
    let b = [8, 10, 7, 9, 6];
    let p = [2, 4, 1, 3, 0];

    ipermute(&mut x, &b, &p);

    assert_eq!(x, [6, 7, 8, 9, 10]);
}

#[test]
fn test_sortperm() {
    let mut p = vec![0usize; 5];
    let v: Vec<isize> = vec![10, 4, -3, 8, -5];
    let vsorted: Vec<isize> = vec![-5, -3, 4, 8, 10];
    sortperm(&mut p, &v);

    for i in 0..v.len() {
        assert_eq!(v[p[i]], vsorted[i]);
    }
}

#[test]
fn test_sortperm_by() {
    let mut p = vec![0usize; 6];
    let v: Vec<isize> = vec![3, 1, 1, 5, 0, 6];
    let ptarg: Vec<usize> = vec![5, 3, 0, 1, 2, 4];

    // sort in descending order should preserve the
    // order of the 1's.   This is different from applying
    // sortperm and then reversing the result.
    sortperm_by(&mut p, &v, |&a, &b| b.cmp(&a));

    for i in 0..v.len() {
        assert_eq!(p[i], ptarg[i]);
    }
}
