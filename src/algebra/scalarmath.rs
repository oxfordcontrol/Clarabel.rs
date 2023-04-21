use super::{FloatT, ScalarMath};

impl<T: FloatT> ScalarMath for T {
    type T = T;
    fn clip(&self, min_thresh: T, max_thresh: T, min_new: T, max_new: T) -> T {
        if *self < min_thresh {
            min_new
        } else if *self > max_thresh {
            max_new
        } else {
            *self
        }
    }

    fn logsafe(&self) -> T {
        if *self <= T::zero() {
            -T::infinity()
        } else {
            self.ln()
        }
    }
}

pub(crate) fn triangular_number(k: usize) -> usize {
    (k * (k + 1)) >> 1
}

#[cfg(feature = "sdp")]
pub(crate) fn triangular_index(k: usize) -> usize {
    // 0-based index into a packed triangle. Same as:
    // triangular number(k+1) - 1 = (((k+1) * (k+2)) >> 1) - 1
    (k * (k + 3)) >> 1
}
