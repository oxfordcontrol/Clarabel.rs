#![allow(non_snake_case)]
use core::cmp::{min,max};
use clarabel_algebra::*;
use derive_builder::Builder;

#[derive(Builder,Debug)]
pub struct QDLDLSettings<T: FloatT> {
    #[builder(default = "None",setter(strip_option))]
    perm: Option<Vec<usize>>,
    #[builder(default = "false")]
    logical: bool,
    #[builder(default = "None",setter(strip_option))]
    Dsigns: Option<Vec<i8>>,
    #[builder(default = "true")]
    regularize_enable: bool,
    #[builder(default = "T::from(1e-12).unwrap()")]
    regularize_eps: T,
    #[builder(default = "T::from(1e-7).unwrap()")]
    regularize_delta: T,
}

impl<T:FloatT> Default for QDLDLSettings<T>{
    fn default() -> QDLDLSettings<T>{
        QDLDLSettingsBuilder::<T>::default().build().unwrap()
    }
}

#[derive(Debug)]
pub struct QDLDLFactorisation<T: FloatT = f64> {

    // permutation vector
    perm: Vec<usize>,
    // inverse permutation
    iperm: Vec<usize>,
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

    pub fn new(Ain: &CscMatrix<T>,
               opts: Option<QDLDLSettings<T>>)
               -> QDLDLFactorisation<T> {_qdldl_new(Ain,opts)}

    pub fn positive_inertia(&self)->usize {self.workspace.positive_inertia}
    pub fn regularize_count(&self)->usize {self.workspace.regularize_count}

    // Solves Ax = b using LDL factors for A.
    // Solves in place (x replaces b)
    pub fn solve(&mut self, b: &mut [T]){

        // bomb if logical factorisation only
        if self.is_logical {
            panic!("Can't solve with logical factorisation only");
        }

        // bomb if b is the wrong size
        assert_eq!(b.len(),self.D.len());

        // permute b
        let tmp = &mut self.workspace.fwork;
        _permute(tmp, b, &self.perm);

        //solve in place with tmp as permuted RHS
        _solve(&self.L.colptr,
               &self.L.rowval,
               &self.L.nzval,
               &self.Dinv,
               tmp);

        // inverse permutation to put unpermuted soln in b
        _ipermute(b,tmp,&self.perm);
    }

    pub fn update_values(&mut self,indices: &[usize], values: &[T]){

        let nzval   = &mut self.workspace.triuA.nzval; // post perm internal data
        let AtoPAPt = &self.workspace.AtoPAPt;         //mapping from input matrix entries to triuA

        for (i,idx) in indices.iter().enumerate() {
            nzval[AtoPAPt[*idx]] = values[i];
        }
    }

    pub fn offset_values(&mut self, indices: &[usize], offset: T){

        let nzval   = &mut self.workspace.triuA.nzval;     //post permutation internal data
        let AtoPAPt = &self.workspace.AtoPAPt;       //mapping from input matrix entries to triuA
        let Dsigns   = &self.workspace.Dsigns;

        for (i,idx) in indices.iter().enumerate() {
            let sign : T = T::from_i8(Dsigns[i]).unwrap();
            nzval[AtoPAPt[*idx]] += offset*sign;
        }
    }

    pub fn update_diagonal(&mut self, indices: &[usize], values: &[T]){

        if !(indices.len() == values.len() || values.len() == 1){
            panic!("Index and value arrays must be the same size, or values must be an array of length 1.");
        }

        let triuA = &mut self.workspace.triuA;
        let invp  = &self.iperm;
        let nvals = values.len();

        // triuA should be full rank and upper triangular, so the diagonal element
        // in each column should always be the last nonzero
        for (i,idx) in indices.iter().enumerate() {
            let thecol = invp[*idx];
            let elidx  = triuA.colptr[thecol+1]; //first element in the *next* column
            if elidx == 0 {panic!("triu(A) is missing diagonal entries");}
            let elidx = elidx - 1;
            let therow = triuA.rowval[elidx];
            if therow != thecol {panic!("triu(A) is missing diagonal entries");}
            let val = if nvals == 1 {values[0]} else {values[i]};
            triuA.nzval[elidx] = val;
        }
    }

    pub fn refactor(&mut self){
        // It never makes sense to call refactor for a logical
        // factorization since it will always be the same.  Calling
        // this function implies that we want a numerical factorization
        self.is_logical = false;
        _factor(
            &mut self.L,
            &mut self.D,
            &mut self.Dinv,
            &mut self.workspace,
            self.is_logical);
    }
}


fn _qdldl_new<T: FloatT> (
    Ain: &CscMatrix<T>,
    opts: Option<QDLDLSettings<T>>) -> QDLDLFactorisation<T>
{

    assert_eq!(Ain.nrows(),Ain.ncols());
    let n = Ain.nrows();

    //get default values if no options passed at all
    let opts = opts.unwrap_or_default();

    //Use AMD ordering if a user-provided ordering
    //is not supplied.   For no ordering at all, the
    //user would need to pass (0..n).collect() explicitly
    let (perm,iperm);
    if let Some(_perm) = opts.perm {
        iperm = _invperm(&_perm);
        perm  = _perm;
    }
    else{
        (perm,iperm) = _get_amd_ordering(Ain);
    }

    //permute to (another) upper triangular matrix and store the
    //index mapping the input's entries to the permutation's entries
    let (A, AtoPAPt) = _permute_symmetric(Ain, &iperm);

    // handle the (possibly permuted) vector of
    // diagonal D signs if one was specified.  Otherwise
    // otherwise all signs are positive
    let mut Dsigns = vec![1_i8; n];
    if let Some(ds) = opts.Dsigns {
        Dsigns = vec![1_i8; n];
        _permute(&mut Dsigns, &ds, &perm);
    }

    // allocate workspace
    let mut workspace = QDLDLWorkspace::<T>::new(
        A,AtoPAPt,Dsigns,opts.regularize_enable,opts.regularize_eps,opts.regularize_delta);

    //total nonzeros in factorization
    let sumLnz = workspace.Lnz.iter().sum();

    // allocate space for the L matrix row indices and data
    let mut L = CscMatrix::spalloc(n,n,sumLnz);

    // allocate for D and D inverse in LDL^T
    let mut D    = vec![T::zero();n];
    let mut Dinv = vec![T::zero();n];

    // factor the matrix into A = LDL^T
    _factor(&mut L, &mut D, &mut Dinv, &mut workspace, opts.logical);

    QDLDLFactorisation{perm, iperm, L, D, Dinv, workspace, is_logical: opts.logical}
}

#[derive(Debug)]
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
    AtoPAPt: Vec<usize>,

    //regularization signs and parameters
    Dsigns: Vec<i8>,
    regularize_enable: bool,
    regularize_eps: T,
    regularize_delta: T,

    // number of regularized entries in D
    regularize_count: usize,
}

impl<T: FloatT> QDLDLWorkspace<T>
{
    pub fn new(
        triuA: CscMatrix<T>,
        AtoPAPt: Vec<usize>,
        Dsigns: Vec<i8>,
        regularize_enable: bool,
        regularize_eps: T,
        regularize_delta: T) -> Self
    {
        let mut etree = vec![0; triuA.ncols()];
        let mut Lnz   = vec![0; triuA.ncols()]; //nonzeros in each L column
        let mut iwork = vec![0; triuA.ncols()*3];
        let bwork = vec![false;triuA.ncols()];
        let fwork = vec![T::zero(); triuA.ncols()];

        // compute elimination tree using QDLDL converted code
        _etree(triuA.nrows(),&triuA.colptr,&triuA.rowval,&mut iwork,&mut Lnz,&mut etree).unwrap();

        // positive inertia count.
        let positive_inertia = 0;

        // number of regularized entries in D. None to start
        let regularize_count = 0;

        Self{etree,Lnz,iwork,bwork,fwork,positive_inertia,triuA,AtoPAPt,Dsigns,regularize_enable,regularize_eps,regularize_delta,regularize_count}
    }
}

fn _factor<T:FloatT>(L: &mut CscMatrix<T>, D: &mut [T], Dinv: &mut [T], workspace: &mut QDLDLWorkspace<T>, logical: bool)
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
        D,
        Dinv,
        &workspace.Lnz,
        &workspace.etree,
        &mut workspace.bwork,
        &mut workspace.iwork,
        &mut workspace.fwork,
        logical,
        &workspace.Dsigns,
        workspace.regularize_enable,
        workspace.regularize_eps,
        workspace.regularize_delta,
        &mut workspace.regularize_count);

    if pos_d_count.is_err() {
        panic!("Zero entry in D (matrix is not quasidefinite)");
    }
    workspace.positive_inertia = pos_d_count.unwrap();

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
    work: &mut [usize],
    Lnz: &mut [usize],
    etree: &mut [usize]) -> Result<usize,i8>
{

    // zero out Lnz and work.  Set all etree values to unknown
    work.fill(0);
    Lnz.fill(0);
    etree.fill(QDLDL_UNKNOWN);

    //Abort if A doesn't have at least one entry in every column
    if Ap.iter()
         .zip(Ap.iter().skip(1))
         .any(|(&current, &next)| current == next)
    {
        return Err(-1);
    }

    // compute the elimination tree
    for j in 0..n {
        work[j] = j;
        for istart in Ai.iter().take(Ap[j+1]).skip(Ap[j]) {

            let mut i = *istart;

            if i > j {return Err(-1);}

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

    Ok(0)
}

//allow too_many_arguments since this follows the implementation
//of the C version of QDLDL.
#[allow(clippy::too_many_arguments)]
fn _factor_inner<T:FloatT>(
        n: usize,
        Ap: &[usize],
        Ai: &[usize],
        Ax: &[T],
        Lp: &mut [usize],
        Li: &mut [usize],
        Lx: &mut [T],
        D: &mut [T],
        Dinv: &mut [T],
        Lnz: &[usize],
        etree: &[usize],
        bwork: &mut [bool],
        iwork: &mut [usize],
        fwork: &mut [T],
        logical_factor: bool,
        Dsigns: &[i8],
        regularize_enable: bool,
        regularize_eps: T,
        regularize_delta: T,
        regularize_count: &mut usize
) -> Result<usize,i8>
{
    *regularize_count = 0;
    let mut positiveValuesInD  = 0;

    // partition working memory into pieces
    let y_markers     = bwork;
    let (y_idx,iwork) = iwork.split_at_mut(n);
    let (elim_buffer,next_colspace) = iwork.split_at_mut(n);
    let y_vals       = fwork;

    //set Lp to cumsum(Lnz), starting from zero
    Lp[0] = 0;
    let mut acc = 0;
    for (Lp, Lnz) in Lp[1..].iter_mut().zip(Lnz){
        *Lp = acc + Lnz;
        acc    = *Lp;
    }

    //  set all y_idx to be 'unused' initially
    // in each column of L, the next available space
    // to start is just the first space in the column
    y_markers.fill(QDLDL_UNUSED);
    y_vals.fill(T::zero());
    D.fill(T::zero());
    next_colspace.copy_from_slice(&Lp[0..Lp.len()-1]);

    if !logical_factor {

        // First element of the diagonal D.
        D[0] = Ax[0];
        if regularize_enable {
            let sign = T::from_i8(Dsigns[0]).unwrap();
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
        let mut nnz_y = 0;  // number of elements in this row

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
                let mut nnz_e      = 1;         //length of unvisited elimination path from here

                let mut next_idx = etree[bidx];

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

                let (f,l) = (Lp[cidx],tmp_idx);
                for (Lxj,Lij) in Lx[f..l].iter().zip(Li[f..l].iter()){
                    y_vals[*Lij] -= (*Lxj)*y_vals_cidx;
                }

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

        // apply dynamic regularization
        if regularize_enable {
            let sign = T::from_i8(Dsigns[k]).unwrap();
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

    Ok(positiveValuesInD)

}


// Solves (L+I)x = b, with x replacing b
fn _lsolve<T:FloatT>(
    Lp: &[usize],
    Li: &[usize],
    Lx: &[T],
    x: &mut[T]

){
    for i in 0..x.len() {
        let xi = x[i];
        let (f,l) = (Lp[i],Lp[i+1]);
        let Lx = &Lx[f..l];
        let Li = &Li[f..l];
        for (Lij,Lxj) in Li.iter().zip(Lx.iter()) {
            x[*Lij] -= (*Lxj)*xi;
        }
    }
}


// Solves (L+I)'x = b, with x replacing b
fn _ltsolve<T: FloatT>(
    Lp: &[usize],
    Li: &[usize],
    Lx: &[T],
    x: &mut[T]
){
    for i in (0..x.len()).rev() {
        let mut s = T::zero();
        let (f,l) = (Lp[i],Lp[i+1]);
        for (Lij,Lxj) in Li[f..l].iter().zip(Lx[f..l].iter()) {
            s += (*Lxj)*x[*Lij];
        }
        x[i] -= s;
    }
}

// Solves Ax = b where A has given LDL factors, with x replacing b
fn _solve<T: FloatT>(
    Lp: &[usize],
    Li: &[usize],
    Lx: &[T],
    Dinv: &[T],
    b: &mut[T]
){
    _lsolve(Lp,Li,Lx, b);
    b.iter_mut().zip(Dinv).for_each(|(b,d)| *b *= *d);
    _ltsolve(Lp,Li,Lx, b);
}

// Construct an inverse permutation from a permutation
fn _invperm(p: &[usize]) -> Vec<usize> {

    let mut b = vec![0; p.len()];

    for (i,j) in p.iter().enumerate() {
        if *j < p.len() && b[*j] == 0 {
            b[*j] = i;
        }
        else{
            panic!("Input vector is not a permutation");
        }
    }
    b
}

// internal permutation and inverse permutation
// functions that require no memory allocations

fn _permute<T: Copy>(x: &mut [T], b: &[T] ,p: &[usize]){
    p.iter().zip(x.iter_mut()).for_each(|(p,x)| *x = b[*p]);
}

fn _ipermute<T: Copy>(x: &mut [T], b: &[T] ,p: &[usize]){
    p.iter().zip(b.iter()).for_each(|(p,b)| x[*p] = *b);
}


// Given a sparse symmetric matrix `A` (with only upper triangular entries), return
// permuted sparse symmetric matrix `P` (also only upper triangular) given the
// inverse permutation vector `iperm`."
fn _permute_symmetric<T: FloatT>(
    A: &CscMatrix<T>,
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
    let mut AtoPAPt = vec![0; A.nnz()];

    _permute_symmetric_inner(A, &mut AtoPAPt, iperm, &mut P.rowval, &mut P.colptr, &mut P.nzval);
    (P, AtoPAPt)
}

// the main function without extra argument checks
// following the book: Timothy Davis - Direct Methods for Sparse Linear Systems

fn _permute_symmetric_inner<T:FloatT>(
    A: &CscMatrix<T>,
    AtoPAPt: &mut [usize],
    iperm: &[usize],
    Pr: &mut [usize],
    Pc: &mut [usize],
    Pv: &mut [T])
{
    // 1. count number of entries that each column of P will have
    let n = A.nrows();
    let mut num_entries = vec![0; n];
    let Ar = &A.rowval;
    let Ac = &A.colptr;
    let Av = &A.nzval;

    // count the number of upper-triangle entries in columns of P,
    // keeping in mind the row permutation
    for colA in 0..n {
        let colP = iperm[colA];
        // loop over entries of A in column A...
        for rowA in Ar.iter().take(Ac[colA+1]).skip(Ac[colA]) {
            let rowP = iperm[*rowA];
            // ...and check if entry is upper triangular
            if *rowA <= colA {
                // determine to which column the entry belongs after permutation
                let col_idx = max(rowP, colP);
                num_entries[col_idx] += 1;
            }
        }
    }

    println!("num_entries = {:?}",num_entries);

    // 2. calculate permuted Pc = P.colptr from number of entries
    // Pc is one longer than num_entries here.
    Pc[0] = 0;
    let mut acc = 0;
    for (Pckp1, ne) in Pc[1..].iter_mut().zip(&num_entries){
        *Pckp1 = acc + ne;
        acc    = *Pckp1;
    }
    // reuse this memory to keep track of free entries in rowval
    num_entries.copy_from_slice(&Pc[0..n]);
    println!("Pc = {:?}",Pc);
    println!("num_entries copied = {:?}",num_entries);

    // use alias
    let mut row_starts = num_entries;
    println!("row_starts copied = {:?}",row_starts);

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

fn _get_amd_ordering<T: FloatT>(A: &CscMatrix<T>) -> (Vec<usize>,Vec<usize>){

    let control = amd::Control::default();
    amd::control(&control);
    let (perm, iperm, _info) =
    amd::order(A.nrows(), &A.colptr, &A.rowval, &control).unwrap();
    (perm,iperm)
}

//configure tests of internals
#[path = "test.rs"]
#[cfg(test)]
mod test;
