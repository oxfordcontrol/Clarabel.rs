use crate::*;

//PJG: Do I really want all fields public here?
//See https://doc.rust-lang.org/reference/visibility-and-privacy.html

#[derive(Debug, Clone)]
pub struct CscMatrix<T: FloatT = f64> {
    pub m: usize,
    pub n: usize,
    pub nzval: Vec<T>,
    pub colptr: Vec<usize>,
    pub rowval: Vec<usize>,
}
