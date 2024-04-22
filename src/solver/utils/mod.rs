//! Internal utility functions and helpers.

use std::iter::zip;

pub(crate) mod atomic;
pub(crate) mod infbounds;

// assorted other functionality missing from std

// a drop-in replacement for the julia "findall" function,
// which serves as a vectorized version of the std::iter::position
// returning indices of *all* elements satisfying a predicate

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
    zip(p, x).for_each(|(p, x)| *x = b[*p]);
}

#[allow(dead_code)]
pub(crate) fn ipermute<T: Copy>(x: &mut [T], b: &[T], p: &[usize]) {
    zip(p, b).for_each(|(p, b)| x[*p] = *b);
}

// -------------
// testing

#[test]
fn test_position_all() {
    let test = [3, 1, 0, 5, 9];
    let idx = test.iter().position_all(|&v| *v > 2);
    assert_eq!(idx, vec![0, 3, 4]);

    let idx = test.iter().position_all(|&v| *v == 2);
    assert_eq!(idx, vec![]);
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
