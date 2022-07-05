#[derive(Debug, Clone)]
pub struct CscMatrix<T = f64> {
    pub m: usize,
    pub n: usize,
    pub colptr: Vec<usize>,
    pub rowval: Vec<usize>,
    pub nzval: Vec<T>,
}
