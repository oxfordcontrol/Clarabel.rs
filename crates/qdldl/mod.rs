use crate::algebra::*;
use core::cmp::{min,max};

pub struct QDLDLSettings<T: FloatT> {
    perm: Option<Vec<usize>>,
    logical: bool,
    Dsigns: Option<Vec<T>>,
    regularize_eps: T,
    regularize_delta: T,
}

impl<T:FloatT> Default for QDLDLSettings<T> {
    fn default() -> Self {
        Self{
            perm: None,
            logical: true,
            Dsigns: None,
            regularize_eps: T::from(1e-12).unwrap(),
            regularize_delta: T::from(1e-7).unwrap()
        }
    }
}

pub fn qdldl<T: FloatT> (
    Ain: &CscMatrix<T>,
    perm: Vec<usize>,
    opts: Option<QDLDLSettings<T>>) -> QDLDLFactorisation<T>
{

    //get default values if no options passed at all
    let opts = opts.unwrap_or_default();

    //store the inverse permutation to enable matrix updates
    let iperm;
    let perm = opts.perm;
    if let Some(perm) = perm {
        iperm = Some(_invperm(&perm));
    }
    else{
        iperm = None;
    }

    // permute using symperm, producing a triu matrix to factors
    let (A, AtoPAPt);
    if let Some(perm) = perm {
        (A, AtoPAPt) = permute_symmetric(Ain, iperm);  //returns an upper triangular matrix
    }
    else{
        (A, AtoPAPt) = (Ain.clone(),None);
    }

    // handle the (possibly permuted) vector
    // diagonal D signs if one was specified
    let mut Dsigns = opts.Dsigns;
    if let Some(ds) = Dsigns {
        if let Some(perm) = perm {
            let mut dsign_perm = vec![T::zero(); ds.len()];
            _permute(&dsign_perm,&Dsigns,&perm);
            Dsigns = Some(dsign_perm);
        }
    }

    // allocate workspace
    let workspace = QDLDLWorkspace::<T>::new(A,AtoPAPt,Dsigns,opts.regularize_eps,opts.regularize_delta);

    // allocate space for the L matrix row indices and data
    let L = CscMatrix::spalloc(A.nrows(),A.nrows(),workspace.Lnz);
    // allocate for D and D inverse
    let D    = vec![T::zero(),A.nrows()];
    let Dinv = vec![T::zero();A.nrows()];

    // factor the matrix
    _factor(perm, &iperm, &L, &D, &Dinv, &workspace, opts.logical);

    return QDLDLFactorisation{perm, iperm, L, D, Dinv, workspace, is_logical: opts.logical}
}

pub struct QDLDLFactorisation<T: FloatT = f64> {

    // permutation vector (nothing if no permutation)
    perm: Option<Vec<usize>>,
    // inverse permutation (nothing if no permutation)
    iperm: Option<Vec<usize>>,
    // lower triangular factor
    L: CscMatrix<T>,
    // D and is inverse for A = LDL^T
    D: Vec<T>,
    Dinv: Vec<T>,
    // workspace data
    workspace: QDLDLWorkspace<T>,
    // is it logical factorisation only?
    is_logical: bool,
}


impl<T:FloatT> QDLDLFactorisation<T>{

    pub fn positive_inertia(&self){self.workspace.positive_inertia}
    pub fn regularize_count(&self){self.workspace.regularize_count}

    // Solves Ax = b using LDL factors for A.
    // Solves in place (x replaces b)
    fn solve(&self, b: &mut [T]){

        // bomb if logical factorisation only
        if self.logical {
            panic!("Can't solve with logical factorisation only");
        }

        // permute b
        let tmp;
        if let perm = Some(self.perm){
            _permute(self.workspace.fwork,b,self.perm);
            tmp = &self.workspace.fwork;
        }
        else{
            tmp = &b;
        }

        _solve(&self.workspace.Ln,
                &self.workspace.Lp,
                &self.workspace.Li,
                &self.workspace.Lx,
                &self.workspace.Dinv,
                &tmp);

        // inverse permutation
        if let perm = Some(self.perm){
            _ipermute(b,self.workspace.fwork,self.perm);
        }
    }

    pub fn update_values(&mut self,indices: &[usize], values: &[usize]){

        let vals    = &self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt;     //mapping from input matrix entries to triuA
        for (i,idx) in indices.iter().enumerate() {
            vals[AtoPAPt[idx]] = values[i];
        }
    }

    pub fn offset_values(&self, indices: &[usize], offset: &[T], signs: Option<&[i8]>){

        let triuA   = &self.workspace.triuA;     //post permutation internal data
        let AtoPAPt = &self.workspace.AtoPAPt;   //mapping from input matrix entries to triuA

        if let Some(signs) = signs {
            for idx in indices.iter(){
                triuA.nzval[AtoPAPt[idx]] += offset;
            }
        }
        else {
            for (i,idx) in indices.iter().enumerate() {
                triuA.nzval[AtoPAPt[idx]] += offset*signs[i];
            }
        }
    }

    pub fn update_diagonal(&self, indices: &[usize], values: &[T]){

        if !(indices.len() == values.len() || values.len() == 1){
            panic!("Index and value arrays must be the same size, or values must be an array of length 1.");
        }

        let triuA = self.workspace.triuA;
        let invp  = self.iperm;
        let nvals = values.len();

        // triuA should be full rank and upper triangular, so the diagonal element
        // in each column should always be the last nonzero
        for (i,idx) in indices.iter().enumerate() {
            let thecol = invp[idx];
            let elidx  = triuA.colptr[thecol+1]; //first element in the *next* column
            if elidx == 0 {panic!("triu(A) is missing diagonal entries");}
            let elidx = elidx - 1;
            let therow = triuA.rowval[elidx];
            if !(therow == thecol) {panic!("triu(A) is missing diagonal entries");}
            let val = if nvals == 1 {values[0]} else {values[i]};
            triuA.nzval[elidx] = val;
        }
    }

    pub fn refactor(&self){
        // It never makes sense to call refactor for a logical
        // factorization since it will always be the same.  Calling
        // this function implies that we want a numerical factorization
        self.logical = false;
        self.workspace.factor(self.logical);
    }
}


pub struct QDLDLWorkspace<T: FloatT = f64> {

    // internal workspace data
    etree: Vec<usize>,
    Lnz: Vec<usize>,
    iwork: Vec<usize>,
    bwork: Vec<bool>,
    fwork: Vec<T>,

    // number of positive values in D
    positive_inertia: usize,

    // The upper triangular matrix factorisation target
    // This is the post ordering PAPt of the original data
    triuA: CscMatrix<T>,

    // mapping from entries in the triu form
    // of the original input to the post ordering
    // triu form used for the factorization
    // this can be used when modifying entries
    // of the data matrix for refactoring
    AtoPAPt: Option<Vec<usize>>,

    //regularization signs and parameters
    Dsigns: Option<Vec<i8>>,
    regularize_eps: T,
    regularize_delta: T,

    // number of regularized entries in D
    regularize_count: usize,
}

impl<T: FloatT> QDLDLWorkspace<T>
{
    pub fn new(
        triuA: CscMatrix<T>,
        AtoPAPt: Option<Vec<usize>>,
        Dsigns: Option<Vec<i8>>,
        regularize_eps: T,
        regularize_delta: T) -> Self
    {
        let etree = vec![0; triuA.ncols()];
        let Lnz   = vec![0; triuA.ncols()]; //nonzeros in each L column
        let iwork = vec![0; triuA.ncols()*3];
        let bwork = vec![false;triuA.ncols()];
        let fwork = vec![0; triuA.ncols()];

        // compute elimination tree using QDLDL converted code
        let sumLnz = _etree(triuA.nrows(),&triuA.colptr,&triuA.rowval,&iwork,&Lnz,&etree);

        if Err(sumLnz) {
            panic!("Input matrix is not upper triangular or has an empty column");
        }

        // positive inertia count.
        let positive_inertia = 0;

        // number of regularized entries in D. None to start
        let regularize_count = 0;

        Self{etree,iwork,bwork,fwork,positive_inertia,triuA,AtoPAPt,Dsigns,regularize_eps,regularize_delta,regularize_count}
    }
}

fn _factor<T:FloatT>(L: &mut CscMatrix<T>, D: &mut [T], Dinv: &mut [T], workspace: &QDLDLWorkspace<T>, logical: bool)
{
    if logical {
        L.nzval.fill(T::zero());
        D.fill(T::zero());
        Dinv.fill(T::zero());
    }

    // factor using QDLDL C style converted code
    let A = &workspace.triuA;

    let pos_d_count =_factor_inner(
        A.n,
        &A.colptr,
        &A.rowval,
        &A.nzval,
        &mut L.colptr,
        &mut L.rowval,
        &mut L.nzval,
        &mut D,
        &mut Dinv,
        &workspace.Lnz,
        &workspace.etree,
        &workspace.bwork,
        &workspace.iwork,
        &workspace.fwork,
        logical,
        &workspace.Dsigns,
        workspace.regularize_eps,
        workspace.regularize_delta,
        &workspace.regularize_count);

    if pos_d_count.is_err() {
        panic!("Zero entry in D (matrix is not quasidefinite)");
    }
    workspace.positive_inertia = Some(pos_d_count.ok());

}


const QDLDL_UNKNOWN: usize = usize::MAX;
const QDLDL_USED: bool   = true;
const QDLDL_UNUSED: bool = false;

// Compute the elimination tree for a quasidefinite matrix
// in compressed sparse column form.

fn _etree(
    n:usize,
    Ap: &[usize],
    Ai: &[usize],
    work: &[usize],
    Lnz: &[usize],
    etree: &[usize]) -> Result<usize,i8>
{

    // zero out Lnz and work.  Set all etree values to unknown
    work.fill(0);
    Lnz.fill(0);
    etree.fill(QDLDL_UNKNOWN);

    //Abort if A doesn't have at least one entry in every column
    if Ap.iter()
         .zip(Ap.iter().skip(1))
         .all(|(current, next)| current != next)
    {return Err(-1 as i8);}

    // compute the elimination tree
    for j in 0..n {
        work[j] = j;
        for p in Ap[j]..Ap[j+1] {

            let i = Ai[p];
            if i > j {return    Err(-1 as i8);}

            while work[i] != j {
                if etree[i] == QDLDL_UNKNOWN {
                    etree[i] = j;
                }
                Lnz[i] += 1;       // nonzeros in this column
                work[i] = j;
                i = etree[i];
            }
        }
    }

    // tally the nonzeros
    let sumLnz = Lnz.iter().sum();
    Ok(sumLnz)
}


fn _factor_inner<T:FloatT>(
        n: usize,
        Ap: &[usize],
        Ai: &[usize],
        Ax: &[T],
        Lp: &[usize],
        Li: &[usize],
        Lx: &mut [T],
        D: &mut [T],
        Dinv: &mut [T],
        Lnz: &[usize],
        etree: &[usize],
        bwork: &[bool],
        iwork: &[usize],
        fwork: &[T],
        logical_factor: bool,
        Dsigns: Option<&[T]>,
        regularize_eps: T,
        regularize_delta: T,
        regularize_count: &usize
) -> Result<usize,i8>
{
    *regularize_count = 0;
    let positiveValuesInD  = 0;

    // partition working memory into pieces
    let y_markers    = bwork;
    let (y_idx,iwork) = iwork.split_at_mut(n);
    let (elim_buffer,next_colspace) = iwork.split_at_mut(n);
    let y_vals       = fwork;

    //set Lp to cumsum(Lnz), starting from zero
    Lp.iter_mut().fold(0, |acc, x| {*x += acc; *x});

    //  set all y_idx to be 'unused' initially
    // in each column of L, the next available space
    // to start is just the first space in the column
    y_markers.fill(QDLDL_UNUSED);
    y_vals.fill(T::zero());
    D.fill(T::zero());
    next_colspace.copy_from_slice(&Lp);

    if !logical_factor {

        // First element of the diagonal D.
        D[0]     = Ax[0];
        if let Some(Dsigns) = Dsigns {
            let sign: T = Dsigns[0];
            if D[0]*sign < regularize_eps {
                D[0] = regularize_delta * sign;
                *regularize_count += 1;
            }
        }

        if D[0] == T::zero() {return Err(-1);}
        if D[0]  > T::zero() {positiveValuesInD += 1;}
        Dinv[0] = T::recip(D[0]);
    }

    // Start from second row (k=1) here. The upper LH corner is trivially 0
    // in L b/c we are only computing the subdiagonal elements
    for k in 1..n {

        // NB : For each k, we compute a solution to
        // y = L(0:(k-1),0:k-1))\b, where b is the kth
        // column of A that sits above the diagonal.
        // The solution y is then the kth row of L,
        // with an implied '1' at the diagonal entry.

        // number of nonzeros in this row of L
        let nnz_y = 0;  // number of elements in this row

        // This loop determines where nonzeros
        // will go in the kth row of L, but doesn't
        // compute the actual values

        for i in Ap[k]..Ap[k+1]{

            let bidx = Ai[i];   //we are working on this element of b

            // Initialize D[k] as the element of this column
            // corresponding to the diagonal place.  Don't use
            // this element as part of the elimination step
            // that computes the k^th row of L
            if bidx == k {
                D[k] = Ax[i];
                continue;
            }

            y_vals[bidx] = Ax[i];   // initialise y(bidx) = b(bidx)

            // use the forward elimination tree to figure
            // out which elements must be eliminated after
            // this element of b
            let next_idx = bidx;

            if y_markers[next_idx] == QDLDL_UNUSED {  //this y term not already visited

                y_markers[next_idx] = QDLDL_USED;     //I touched this one
                elim_buffer[0]     = next_idx;  // It goes at the start of the current list
                let nnz_e          = 1;         //length of unvisited elimination path from here

                let next_idx = etree[bidx];

                while next_idx != QDLDL_UNKNOWN && next_idx < k {

                    if y_markers[next_idx] == QDLDL_USED {break;}

                    y_markers[next_idx] = QDLDL_USED;   // I touched this one
                    elim_buffer[nnz_e] = next_idx; // It goes in the current list
                    next_idx = etree[next_idx];   // one step further along tree
                    nnz_e += 1;                   // the list is one longer than before

                }

                // now put the buffered elimination list into
                // my current ordering in reverse order
                while nnz_e != 0 {
                    nnz_e -= 1;
                    y_idx[nnz_y] = elim_buffer[nnz_e];
                    nnz_y += 1;
                }
            }
        }

        // This for loop places nonzeros values in the k^th row
        for i in (0..nnz_y).rev() {

            // which column are we working on?
            let cidx = y_idx[i];

            // loop along the elements in this
            // column of L and subtract to solve to y
            let tmp_idx = next_colspace[cidx];

            // don't compute Lx for logical factorisation
            // this logic is not implemented in the C version
            if !logical_factor {

                let y_vals_cidx = y_vals[cidx];

                let rng = Lp[cidx]..(tmp_idx-1);
                Lx[rng].iter().zip(Li[rng].iter()).for_each(|(Lxj,Lij)|
                    {
                        y_vals[*Lij] -= (*Lxj)*y_vals_cidx;
                    }
                );

                // Now I have the cidx^th element of y = L\b.
                // so compute the corresponding element of
                // this row of L and put it into the right place
                Lx[tmp_idx] = y_vals_cidx * Dinv[cidx];
                D[k] -= y_vals_cidx*Lx[tmp_idx];
            }

            // record which row it went into
            Li[tmp_idx] = k;
            next_colspace[cidx] += 1;

            // reset the y_vals and indices back to zero and QDLDL_UNUSED
            // once I'm done with them
            y_vals[cidx]     = T::zero();
            y_markers[cidx]  = QDLDL_UNUSED;

        }

        // apply dynamic regularization if a sign
        // vector has been specified.
        if let Some(Dsigns) = Dsigns {
            let sign: T = Dsigns[k];
            if D[k]*sign < regularize_eps {
                D[k] = regularize_delta * sign;
                *regularize_count += 1;
            }
        }

        // Maintain a count of the positive entries
        // in D.  If we hit a zero, we can't factor
        // this matrix, so abort
        if D[k] == T::zero()  {return Err(-1);}
        if D[k]  > T::zero()  {positiveValuesInD += 1;}

        // compute the inverse of the diagonal
        Dinv[k] = T::recip(D[k]);

    } //end for k

    return Ok(positiveValuesInD);

}


// Solves (L+I)x = b, with x replacing b
fn _lsolve<T:FloatT>(
    n: usize,
    Lp: &[usize],
    Li: &[usize],
    Lx: &[T],
    x: &mut[T]

){
    for (i,xi) in x.iter().enumerate() {
        let rng = Lp[i]..Lp[i+1];
        for (Lij,Lxj) in Li[rng].iter().zip(Lx[rng].iter()) {
            x[*Lij] -= (*Lxj)*(*xi);
        }
    }
}


// Solves (L+I)'x = b, with x replacing b
fn _ltsolve<T: FloatT>(
    n: usize,
    Lp: &[usize],
    Li: &[usize],
    Lx: &[T],
    x: &mut[T]
){
    for (i,xi) in x.iter().enumerate().rev(){
        let s = T::zero();
        let rng = Lp[i]..Lp[i+1];
        for (Lij,Lxj) in Li[rng].iter().zip(Lx[rng].iter()) {
            s += (*Lxj)*x[*Lij];
        }
        *xi -= s;
    }
}

// Solves Ax = b where A has given LDL factors, with x replacing b
fn _solve<T: FloatT>(
    n: usize,
    Lp: &[usize],
    Li: &[usize],
    Lx: &[T],
    Dinv: &[T],
    b: &mut[T]
){
    _lsolve(n,&Lp,&Li,&Lx,&mut b);
    b.iter_mut().zip(Dinv).for_each(|(b,d)| *b *= *d);
    _ltsolve(n,&Lp,&Li,&Lx,&mut b);
}

// Construct an inverse permutation from a permutation
fn _invperm(p: &[usize]) -> Vec<usize> {

    let b = vec![0 as usize; p.len()];

    for (i,j) in p.iter().enumerate() {
        if 0 <= *j && *j < p.len() && b[*j] == 0 {
            b[*j] = i;
        }
        else{
            panic!("Input vector is not a permutation");
        }
    }
    return b;
}

// internal permutation and inverse permutation
// functions that require no memory allocations

fn _permute<T>(x: &mut [T], b: &[T] ,p: &[usize]){
    p.iter().zip(x).for_each(|(p,x)| *x = b[*p]);
}

fn _ipermute<T>(x: &mut [T], b: &[T] ,p: &[usize]){
    p.iter().zip(b).for_each(|(p,b)| x[*p] = *b);
}


// Given a sparse symmetric matrix `A` (with only upper triangular entries), return
// permuted sparse symmetric matrix `P` (also only upper triangular) given the
// inverse permutation vector `iperm`."
fn permute_symmetric<T: FloatT>(
    A: CscMatrix<T>,
    iperm: &[usize]) -> (CscMatrix<T>, Vec<usize>)
{
    // perform a number of argument checks
    let (m, n) = (A.nrows(),A.ncols());
    if m != n {panic!("Matrix A must be sparse and square")};

    if n != iperm.len(){
        panic!("Dimensions of sparse matrix A must equal the length of iperm");
    }

    let mut P = CscMatrix::<T>::spalloc(n,n,A.nnz());

    // we will record a mapping of entries from A to PAPt
    let AtoPAPt = vec![0; A.nnz()];

    _permute_symmetric(&A, &AtoPAPt, &iperm, &mut P.rowval, &mut P.colptr, &mut P.nzval);
    return (P, AtoPAPt)
}

// the main function without extra argument checks
// following the book: Timothy Davis - Direct Methods for Sparse Linear Systems

fn _permute_symmetric<T:FloatT>(
    A: &CscMatrix<T>,
    AtoPAPt: &[usize],
    iperm: &[usize],
    Pr: &mut [usize],
    Pc: &mut [usize],
    Pv: &mut [T])
{
    // 1. count number of entries that each column of P will have
    let n = A.nrows();
    let num_entries = vec![0; n];
    let Ar = A.rowval;
    let Ac = A.colptr;
    let Av = A.nzval;

    // count the number of upper-triangle entries in columns of P,
    // keeping in mind the row permutation
    for colA in 0..n {
        let colP = iperm[colA];
        // loop over entries of A in column A...
        for row_idx in Ac[colA]..Ac[colA+1] {
            let rowA = Ar[row_idx];
            let rowP = iperm[rowA];
            // ...and check if entry is upper triangular
            if rowA <= colA {
                // determine to which column the entry belongs after permutation
                let col_idx = max(rowP, colP);
                num_entries[col_idx] += 1;
            }
        }
    }

    // 2. calculate permuted Pc = P.colptr from number of entries
    // Pc is one longer than num_entries here.
    Pc[0] = 1;
    let acc = 0;
    for (Pckp1, ne) in Pc[1..].iter_mut().zip(num_entries){
        *Pckp1 = acc + ne;
        acc    = *Pckp1;
    }
    // reuse this vector memory to keep track of free entries in rowval
    num_entries.copy_from_slice(&Pc[0..num_entries.len()]);

    // use alias
    let row_starts = num_entries;

    // 3. permute the row entries and position of corresponding nzval
    for colA in 0..n {
        let colP = iperm[colA];
        // loop over rows of A and determine where each row entry of A should be stored
        for rowA_idx in Ac[colA]..Ac[colA+1] {
            let rowA = Ar[rowA_idx];
            // check if upper triangular
            if rowA <= colA {
                let rowP = iperm[rowA];
                // determine column to store the entry
                let col_idx = max(colP, rowP);

                // find next free location in rowval (this results in unordered columns in the rowval)
                let rowP_idx = row_starts[col_idx];

                // store rowval and nzval
                Pr[rowP_idx] = min(colP, rowP);
                Pv[rowP_idx] = Av[rowA_idx];

                //record this into the mapping vector
                AtoPAPt[rowA_idx] = rowP_idx;

                // increment next free location
                row_starts[col_idx] += 1;
            }
        }
    }

}
