#![allow(non_snake_case)]

use super::*;
use crate::solver::core::cones::*;
use enum_dispatch::*;

#[enum_dispatch(SparseExpansionMapTrait)]
pub(crate) enum SparseExpansionMap {
    SOCExpansionMap(SOCExpansionMap),
    GenPowExpansionMap(GenPowExpansionMap),
}

#[enum_dispatch(SparseExpansionConeTrait<T>)]
pub(crate) enum SparseExpansionCone<'a, T>
where
    T: FloatT,
{
    SecondOrderCone(&'a SecondOrderCone<T>),
    GenPowerCone(&'a GenPowerCone<T>),
}

impl<'a, T> SupportedCone<T>
where
    T: FloatT,
{
    pub(crate) fn to_sparse_expansion(&'a self) -> Option<SparseExpansionCone<T>> {
        match self {
            SupportedCone::SecondOrderCone(sc) => Some(SparseExpansionCone::SecondOrderCone(sc)),
            SupportedCone::GenPowerCone(sc) => Some(SparseExpansionCone::GenPowerCone(sc)),
            _ => None,
        }
    }
}

#[enum_dispatch]
pub(crate) trait SparseExpansionMapTrait {
    fn pdim(&self) -> usize;
    fn nnz_vec(&self) -> usize;
    fn Dsigns(&self) -> &[i8];
}

impl SparseExpansionMapTrait for Vec<SparseExpansionMap> {
    fn pdim(&self) -> usize {
        self.iter().fold(0, |pdim, map| pdim + map.pdim())
    }
    fn nnz_vec(&self) -> usize {
        self.iter().fold(0, |nnz, map| nnz + map.nnz_vec())
    }
    fn Dsigns(&self) -> &[i8] {
        unreachable!()
    }
}

type UpdateFcn<T> = fn(&mut BoxedDirectLDLSolver<T>, &mut CscMatrix<T>, &[usize], &[T]) -> ();
type ScaleFcn<T> = fn(&mut BoxedDirectLDLSolver<T>, &mut CscMatrix<T>, &[usize], T) -> ();

#[enum_dispatch]
pub(crate) trait SparseExpansionConeTrait<T>
where
    T: FloatT,
{
    fn expansion_map(&self) -> SparseExpansionMap;
    fn csc_colcount_sparsecone(
        &self,
        map: &SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    );
    fn csc_fill_sparsecone(
        &self,
        map: &mut SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    );
    fn csc_update_sparsecone(
        &self,
        map: &SparseExpansionMap,
        ldl: &mut BoxedDirectLDLSolver<T>,
        K: &mut CscMatrix<T>,
        updateFcn: UpdateFcn<T>,
        scaleFcn: ScaleFcn<T>,
    );
}

macro_rules! impl_map_recover {
    ($CONE:ident,$MAP:ident) => {
        impl<'a, T> $CONE<T> {
            pub(crate) fn recover_map(&self, map: &'a SparseExpansionMap) -> &'a $MAP {
                match map {
                    SparseExpansionMap::$MAP(map) => map,
                    _ => panic!(),
                }
            }
            pub(crate) fn recover_map_mut(&self, map: &'a mut SparseExpansionMap) -> &'a mut $MAP {
                match map {
                    SparseExpansionMap::$MAP(map) => map,
                    _ => panic!(),
                }
            }
        }
    };
}

//--------------------------------------
// Second order cone data map
//--------------------------------------

pub(crate) struct SOCExpansionMap {
    u: Vec<usize>, //off diag dense columns u
    v: Vec<usize>, //off diag dense columns v
    D: [usize; 2], //diag D
}

impl SOCExpansionMap {
    pub fn new<T: FloatT>(cone: &SecondOrderCone<T>) -> Self {
        let u = vec![0; cone.numel()];
        let v = vec![0; cone.numel()];
        let D = [0; 2];
        Self { u, v, D }
    }
}

impl SparseExpansionMapTrait for SOCExpansionMap {
    fn pdim(&self) -> usize {
        2
    }
    fn nnz_vec(&self) -> usize {
        2 * self.v.len()
    }
    fn Dsigns(&self) -> &[i8] {
        &[-1, 1]
    }
}

impl_map_recover!(SecondOrderCone, SOCExpansionMap);

impl<'a, T> SparseExpansionConeTrait<T> for &'a SecondOrderCone<T>
where
    T: FloatT,
{
    fn expansion_map(&self) -> SparseExpansionMap {
        SparseExpansionMap::SOCExpansionMap(SOCExpansionMap::new(self))
    }

    fn csc_colcount_sparsecone(
        &self,
        map: &SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map(map);
        let nvars = self.numel();

        match shape {
            MatrixTriangle::Triu => {
                K.colcount_colvec(nvars, row, col); // u column
                K.colcount_colvec(nvars, row, col + 1); // v column
            }
            MatrixTriangle::Tril => {
                K.colcount_rowvec(nvars, col, row); // u row
                K.colcount_rowvec(nvars, col + 1, row); // v row
            }
        }
        K.colcount_diag(col, map.pdim());
    }

    fn csc_fill_sparsecone(
        &self,
        map: &mut SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map_mut(map);

        // fill structural zeros for u and v columns for this cone
        // note v is the first extra row/column, u is second
        match shape {
            MatrixTriangle::Triu => {
                K.fill_colvec(&mut map.v, row, col); //u
                K.fill_colvec(&mut map.u, row, col + 1); //v
            }
            MatrixTriangle::Tril => {
                K.fill_rowvec(&mut map.v, col, row); //u
                K.fill_rowvec(&mut map.u, col + 1, row); //v
            }
        }
        let pdim = map.pdim();
        K.fill_diag(&mut map.D, col, pdim);
    }

    fn csc_update_sparsecone(
        &self,
        map: &SparseExpansionMap,
        ldl: &mut BoxedDirectLDLSolver<T>,
        K: &mut CscMatrix<T>,
        updateFcn: UpdateFcn<T>,
        scaleFcn: ScaleFcn<T>,
    ) {
        let sparse_data = self.sparse_data.as_ref().unwrap();

        let map = self.recover_map(map);
        let η2 = self.η * self.η;

        // off diagonal columns (or rows)
        updateFcn(ldl, K, &map.u, &sparse_data.u);
        updateFcn(ldl, K, &map.v, &sparse_data.v);
        scaleFcn(ldl, K, &map.u, -η2);
        scaleFcn(ldl, K, &map.v, -η2);

        //set diagonal to η^2*(-1,1) in the extended rows/cols
        updateFcn(ldl, K, &map.D, &[-η2, η2]);
    }
}

//--------------------------------------
// Generalized power cone data map
//--------------------------------------

pub(crate) struct GenPowExpansionMap {
    p: Vec<usize>, //off diag dense columns p
    q: Vec<usize>, //off diag dense columns q
    r: Vec<usize>, //off diag dense columns r
    D: [usize; 3], //diag D
}

impl GenPowExpansionMap {
    pub fn new<T: FloatT>(cone: &GenPowerCone<T>) -> Self {
        let p = vec![0; cone.numel()];
        let q = vec![0; cone.dim1()];
        let r = vec![0; cone.dim2()];
        let D = [0; 3];
        Self { p, q, r, D }
    }
}

impl SparseExpansionMapTrait for GenPowExpansionMap {
    fn pdim(&self) -> usize {
        3
    }
    fn nnz_vec(&self) -> usize {
        self.p.len() + self.q.len() + self.r.len()
    }
    fn Dsigns(&self) -> &[i8] {
        &[-1, -1, 1]
    }
}

impl_map_recover!(GenPowerCone, GenPowExpansionMap);

impl<'a, T> SparseExpansionConeTrait<T> for &'a GenPowerCone<T>
where
    T: FloatT,
{
    fn expansion_map(&self) -> SparseExpansionMap {
        SparseExpansionMap::GenPowExpansionMap(GenPowExpansionMap::new(self))
    }

    fn csc_colcount_sparsecone(
        &self,
        map: &SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map(map);
        let nvars = self.numel();
        let dim1 = self.dim1();
        let dim2 = self.dim2();

        match shape {
            MatrixTriangle::Triu => {
                K.colcount_colvec(dim1, row, col); //q column
                K.colcount_colvec(dim2, row + dim1, col + 1); //r column
                K.colcount_colvec(nvars, row, col + 2); //p column
            }
            MatrixTriangle::Tril => {
                K.colcount_rowvec(dim1, col, row); //q row
                K.colcount_rowvec(dim2, col + 1, row + dim1); //r row
                K.colcount_rowvec(nvars, col + 2, row); //p row
            }
        }
        K.colcount_diag(col, map.pdim());
    }

    fn csc_fill_sparsecone(
        &self,
        map: &mut SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map_mut(map);
        let dim1 = self.dim1();

        match shape {
            MatrixTriangle::Triu => {
                K.fill_colvec(&mut map.q, row, col); //q column
                K.fill_colvec(&mut map.r, row + dim1, col + 1); //r column
                K.fill_colvec(&mut map.p, row, col + 2); //p column
            }
            MatrixTriangle::Tril => {
                K.fill_rowvec(&mut map.q, col, row); //q row
                K.fill_rowvec(&mut map.r, col + 1, row + dim1); //r row
                K.fill_rowvec(&mut map.p, col + 2, row); //p row
            }
        }
        let pdim = map.pdim();
        K.fill_diag(&mut map.D, col, pdim);
    }

    fn csc_update_sparsecone(
        &self,
        map: &SparseExpansionMap,
        ldl: &mut BoxedDirectLDLSolver<T>,
        K: &mut CscMatrix<T>,
        updateFcn: UpdateFcn<T>,
        scaleFcn: ScaleFcn<T>,
    ) {
        let map = self.recover_map(map);
        let data = &self.data;
        let sqrtμ = data.μ.sqrt();

        //&off diagonal columns (or rows), distribute √μ to off-diagonal terms
        updateFcn(ldl, K, &map.q, &data.q);
        updateFcn(ldl, K, &map.r, &data.r);
        updateFcn(ldl, K, &map.p, &data.p);
        scaleFcn(ldl, K, &map.q, -sqrtμ);
        scaleFcn(ldl, K, &map.r, -sqrtμ);
        scaleFcn(ldl, K, &map.p, -sqrtμ);

        //&normalize diagonal terms to 1/-1 in the extended rows/cols
        updateFcn(ldl, K, &map.D, &[-T::one(), -T::one(), T::one()]);
    }
}

//--------------------------------------
// LDL Data Map
//--------------------------------------

pub(crate) struct LDLDataMap {
    pub P: Vec<usize>,
    pub A: Vec<usize>,
    pub Hsblocks: Vec<usize>, //indices of the lower RHS blocks (by cone)
    pub sparse_maps: Vec<SparseExpansionMap>, //sparse cone expansion terms

    // all of above terms should be disjoint and their union
    // should cover all of the user data in the KKT matrix.  Now
    // we make two last redundant indices that will tell us where
    // the whole diagonal is, including structural zeros.
    pub diagP: Vec<usize>,
    pub diag_full: Vec<usize>,
}

impl LDLDataMap {
    pub fn new<T: FloatT>(
        Pmat: &CscMatrix<T>,
        Amat: &CscMatrix<T>,
        cones: &CompositeCone<T>,
    ) -> Self {
        let (m, n) = (Amat.nrows(), Pmat.nrows());
        let P = vec![0; Pmat.nnz()];
        let A = vec![0; Amat.nnz()];

        // the diagonal of the ULHS KKT block P.
        // NB : we fill in structural zeros here even if the matrix
        // P is empty (e.g. as in an LP), so we can have entries in
        // index Pdiag that are not present in the index P
        let diagP = vec![0; n];

        // make an index for each of the Hs blocks for each cone
        let Hsblocks = allocate_kkt_Hsblocks::<T, usize>(cones);

        // now do the sparse cone expansion pieces
        let nsparse = cones.iter().filter(|&c| c.is_sparse_expandable()).count();
        let mut sparse_maps = Vec::with_capacity(nsparse);

        for cone in cones.iter() {
            if cone.is_sparse_expandable() {
                let sc = cone.to_sparse_expansion().unwrap();
                sparse_maps.push(sc.expansion_map());
            }
        }

        let diag_full = vec![0; m + n + sparse_maps.pdim()];

        Self {
            P,
            A,
            Hsblocks,
            sparse_maps,
            diagP,
            diag_full,
        }
    }
}
