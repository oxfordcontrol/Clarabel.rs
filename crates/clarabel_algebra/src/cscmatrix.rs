// All internal matrix representations in the default 
// solver and math implementations are in standard
// compressed sparse column format, as is the API.

#[derive(Debug, Clone)]
pub struct CscMatrix<T = f64> {
    pub m: usize,
    pub n: usize,
    pub colptr: Vec<usize>,
    pub rowval: Vec<usize>,
    pub nzval: Vec<T>,
}

// Convenience types for marking matrix orientation 
// and sparsity patterns.

// T = transpose, N = non-transpose
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum MatrixShape {
    N,
    T,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum MatrixTriangle {
    Triu,
    Tril,
}