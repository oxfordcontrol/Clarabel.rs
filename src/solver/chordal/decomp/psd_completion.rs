use crate::algebra::*;
use crate::solver::chordal::ChordalInfo;
use crate::solver::chordal::SparsityPattern;
use crate::solver::core::cones::*;
use crate::solver::DefaultVariables;

// -----------------------------------
// psd completion
// -----------------------------------

// The psd entries of z that correspond to the zeros in s are not constrained by the problem.
// however, in order to make the dual psd cone positive semidefinite we have to do a
// positive semidefinite completion routine to choose the values

impl<T> ChordalInfo<T>
where
    T: FloatT,
{
    pub(crate) fn psd_completion(&self, variables: &mut DefaultVariables<T>) {
        // working now with the cones from the original
        // problem, not the decomposed ones
        let cones = &self.init_cones;

        // loop over psd cones
        let row_ranges: Vec<_> = cones.rng_cones_iter().collect();

        // loop over just the patterns
        for pattern in self.spatterns.iter() {
            let row_range = row_ranges[pattern.orig_index].clone();
            let z = &mut variables.z[row_range];
            complete(z, pattern);
        }
    }
}

fn complete<T>(z: &mut [T], pattern: &SparsityPattern)
where
    T: FloatT,
{
    let n = pattern.ordering.len();
    let mut Z = Matrix::zeros((n, n));
    svec_to_mat(&mut Z, z);
    psd_complete(&mut Z, pattern);
    mat_to_svec(z, &Z);
}

// positive semidefinite completion (from Vandenberghe - Chordal Graphs..., p. 362)
// input: A - positive definite completable matrix

fn psd_complete<T>(A: &mut Matrix<T>, pattern: &SparsityPattern)
where
    T: FloatT,
{
    let sntree = &pattern.sntree;
    let p = &pattern.ordering;
    let ip = invperm(p);
    let N = A.ncols();

    // PJG: not clear to me if this copy of A is required, or
    // whether I can operate directly on A by permuting the
    // the indices in the loops below.  Only worth doing that
    // if copying in or out of A is expensive.

    // permutate matrix based on ordering p
    // W is in the order that the cliques are based on
    let mut W = Matrix::zeros((N, N));
    W.subsref(A, p, p);

    // go through supernode tree in descending order (given a post-ordering).
    // This is ensured in the get_snode, get_separators functions

    let mut Wαα = Matrix::<T>::zeros((0, 0));
    let mut Wαν = Matrix::<T>::zeros((0, 0));
    let mut Wηα = Matrix::<T>::zeros((0, 0));
    let mut Wηα_times_Y = Matrix::<T>::zeros((0, 0));

    let mut chol = CholeskyEngine::new(0);
    let mut svd = SVDEngine::new((0, 0));

    for j in (0..(sntree.n_cliques - 1)).rev() {
        // in order to obtain ν, α the vertex numbers of the supernode are
        // mapped to the new position of the permuted matrix index
        // set of snd(i) sorted using the numerical ordering i,i+1,...i+ni
        let ν = sntree.get_snode(j);

        // index set containing the elements of col(i) \ snd(i)
        // sorted using numerical ordering σ(i)
        let α = sntree.get_separators(j);

        // index set containing the row indices of the lower-triangular zeros in
        // column i (i: representative index) sorted by σ(i)
        let i = ν[0];
        let η: Vec<usize> = ((i + 1)..N)
            .filter(|&x| !α.contains(&x) && !ν.contains(&x))
            .collect();

        Wαα.resize((α.len(), α.len()));
        Wαν.resize((α.len(), ν.len()));
        Wηα.resize((η.len(), α.len()));

        Wαα.subsref(&W, α, α);
        Wαν.subsref(&W, α, ν);
        Wηα.subsref(&W, &η, α);

        // Solve WWαα \ Wαν.   First try Cholesky,
        // and then fall back to pinv if it fails.
        // NB: cholesky does not modify the matrix
        // it factors, but the SVD call will
        chol.resize(α.len());
        match chol.factor(&mut Wαα) {
            Ok(()) => {
                chol.solve(&mut Wαν);
            }
            Err(_) => {
                svd.resize((α.len(), α.len()));
                svd.factor(&mut Wαα).unwrap();
                svd.solve(&mut Wαν); //pinv solve
            }
        }

        let Y = &Wαν; //solved in place
        Wηα_times_Y.resize((η.len(), ν.len()));
        Wηα_times_Y.mul(&Wηα, Y, T::one(), T::zero());

        W.subsasgn(&η, ν, &Wηα_times_Y);

        // symmetry condition
        W.subsasgn(ν, &η, &Wηα_times_Y.t());
    }

    // invert the permutation
    A.subsref(&W, &ip, &ip);
}
