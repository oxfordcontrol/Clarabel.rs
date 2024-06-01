use super::{FloatT, ScalarMath};

impl<T: FloatT> ScalarMath for T {
    fn clip(&self, min_thresh: Self, max_thresh: Self) -> Self {
        if *self < min_thresh {
            min_thresh
        } else if *self > max_thresh {
            max_thresh
        } else {
            *self
        }
    }

    fn logsafe(&self) -> Self {
        if *self <= Self::zero() {
            -Self::infinity()
        } else {
            self.ln()
        }
    }
}

pub(crate) fn triangular_number(k: usize) -> usize {
    (k * (k + 1)) >> 1
}

#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) fn triangular_index(k: usize) -> usize {
    // 0-based index into a packed triangle. Same as:
    // triangular number(k+1) - 1 = (((k+1) * (k+2)) >> 1) - 1
    (k * (k + 3)) >> 1
}

// given an index into the upper triangular part of a matrix, return
// its row and column position
#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) fn upper_triangular_index_to_coord(linearidx: usize) -> (usize, usize) {
    if linearidx == 0 {
        return (0, 0);
    }

    let col = ((isqrt(8 * linearidx + 1) + 1) >> 1) - 1;

    let row = linearidx - triangular_index(col - 1) - 1;
    (row, col)
}

// given a row and column position, return the index into the upper
// triangular part of the matrix
#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
pub(crate) fn coord_to_upper_triangular_index(coord: (usize, usize)) -> usize {
    if coord == (0, 0) {
        return 0;
    }

    let (i, j) = coord;
    if i <= j {
        triangular_index(j - 1) + i + 1
    } else {
        triangular_index(i - 1) + j + 1
    }
}

// isqrt : rust does not currently have a stable isqrt yet.   See issue
// here: https://github.com/rust-lang/rust/issues/116226
// For now, implement a simple truncation method, which works
// for inputs < 2^53 or so (otherwise possibly off-by-one).
#[cfg_attr(not(feature = "sdp"), allow(dead_code))]
fn isqrt(v: usize) -> usize {
    (v as f64).sqrt() as usize
}

#[test]
fn test_triangular_number() {
    let v = vec![1, 2, 3, 4, 5, 6, 7];
    let t = vec![1, 3, 6, 10, 15, 21, 28];
    for (ti, vi) in core::iter::zip(t, v) {
        assert_eq!(ti, triangular_number(vi));
    }
}

#[test]
fn test_triangular_index() {
    let v = vec![0, 1, 2, 3, 4, 5, 6];
    let t = vec![0, 2, 5, 9, 14, 20, 27];
    for (ti, vi) in core::iter::zip(t, v) {
        assert_eq!(ti, triangular_index(vi));
    }
}

#[test]
fn test_triangular_index_and_coord() {
    let idx = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let coord = vec![
        (0, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 3),
    ];
    for (i, c) in core::iter::zip(idx, coord) {
        assert_eq!(i, coord_to_upper_triangular_index(c));
        assert_eq!(upper_triangular_index_to_coord(i), c);
    }
}

#[test]
fn test_isqrt() {
    // this test is obviously not very thorough
    assert_eq!(0, isqrt(0));
    assert_eq!(1, isqrt(1));
    assert_eq!(1, isqrt(2));
    assert_eq!(2, isqrt(4));
    assert_eq!(2, isqrt(8));
    assert_eq!(3, isqrt(9));
}
