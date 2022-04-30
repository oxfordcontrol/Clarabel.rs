use super::*;
use crate::algebra::*;
use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;

// -------------------------------------
// Cone Set (default composite cone type)
// -------------------------------------

type BoxedCone<T> = Box<dyn Cone<T>>;

pub struct ConeSet<T: FloatT = f64> {
    cones: Vec<BoxedCone<T>>,

    //Type tags and count of each cone
    pub types: Vec<SupportedCones>,
    pub type_counts: HashMap<SupportedCones, usize>,

    //overall size of the composite cone
    numel: usize,
    degree: usize,

    //ranges for the indices of the constituent cones
    pub rng_cones: Vec<Range<usize>>,

    //ranges for the indices of the constituent WtW blocks
    //associated with each cone
    pub rng_blocks: Vec<Range<usize>>,
}

impl<T: FloatT> ConeSet<T> {
    pub fn new(types: &[SupportedCones], dims: &[usize]) -> Self {

        assert_eq!(types.len(), dims.len());

        // make an internal copy to protect from user modification
        let types = types.to_vec();
        let ncones = types.len();
        let mut cones: Vec<BoxedCone<T>> = Vec::with_capacity(ncones);

        // create cones with the given dims
        for (dim,t) in dims.iter().zip(types.iter()) {
            cones.push(cone_dict(*t, *dim));
        }

        //  count the number of each cone type.
        // PJG: could perhaps fix max capacity here but Enum::variant_count is not
        // yet a stable feature.  Capacity should be number of SupportedCones variants
        let mut type_counts = HashMap::new();
        for t in types.iter() {
            *type_counts.entry(*t).or_insert(0) += 1;
        }

        // count up elements and degree
        let numel = cones.iter().map(|c| c.numel()).sum();
        let degree = cones.iter().map(|c| c.degree()).sum();

        //ranges for the subvectors associated with each cone,
        //and the subranges associated with their corresponding
        //entries in the WtW sparse block

        let rng_cones = _make_rng_cones(&cones);
        let rng_blocks = _make_rng_blocks(&cones);


        Self {
            cones,
            types,
            type_counts,
            numel,
            degree,
            rng_cones,
            rng_blocks,
        }
    }
}

fn _make_rng_cones<T>(cones: &[BoxedCone<T>]) -> Vec<Range<usize>>
where
    T: FloatT,
{
    let mut rngs = Vec::with_capacity(cones.len());

    if !cones.is_empty() {
        let mut start = 0;
        for cone in cones {
            let stop  = start + cone.numel();
            rngs.push(start..stop);
            start = stop;
        }
    }
    rngs
}

fn _make_rng_blocks<T>(cones: &[BoxedCone<T>]) -> Vec<Range<usize>>
where
    T: FloatT,
{
    let mut rngs = Vec::with_capacity(cones.len());

    if !cones.is_empty() {
        let mut start = 0;
        for cone in cones {
            let nvars = cone.numel();
            let stop = start + {
                if cone.WtW_is_diagonal() { nvars }
                else { (nvars * (nvars + 1)) >> 1 }
            };
            rngs.push(start..stop);
            start = stop;
        }
    }
    rngs
}


fn _coneset_make_headidx<T>(headidx: &mut [usize], cones: &[BoxedCone<T>])
where
    T: FloatT,
{
    if !cones.is_empty() {
        // index of first element in each cone
        headidx[0] = 0;
        for i in 2..headidx.len() {
            headidx[i] = headidx[i - 1] + cones[i - 1].numel();
        }
    }
}

impl<T: FloatT> ConeSet<T> {
    pub fn len(&self) -> usize {
        self.cones.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cones.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, BoxedCone<T>> {
        self.cones.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, BoxedCone<T>> {
        self.cones.iter_mut()
    }
    pub fn anyref_by_idx(&self, idx: usize) -> &(dyn Any + '_) {
        &self.cones[idx] as &(dyn Any + '_)
    }
}

impl<T> Cone<T> for ConeSet<T>
where
    T: FloatT,
{
    fn dim(&self) -> usize {
        panic!("dim() not well defined for the ConeSet");
    }

    fn degree(&self) -> usize {
        self.degree
    }

    fn numel(&self) -> usize {
        self.numel
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        let mut any_changed = false;

        // we will update e <- δ .* e using return values
        // from this function.  default is to do nothing at all
        δ.fill(T::one());
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            let δi = &mut δ[rng.clone()];
            let ei = &e[rng.clone()];
            any_changed |= cone.rectify_equilibration(δi, ei);
        }
        any_changed
    }

    fn WtW_is_diagonal(&self) -> bool {
        //This function should probably never be called since
        //we only us it to interrogate the blocks, but we can
        //implement something reasonable anyway
        let mut is_diag = true;
        for cone in self.iter() {
            is_diag &= cone.WtW_is_diagonal();
            if !is_diag {
                break;
            }
        }
        is_diag
    }

    fn update_scaling(&mut self, s: &[T], z: &[T]) {
        let cones = &mut self.cones;
        let rngs  = &self.rng_cones;

        for (cone,rng) in cones.iter_mut().zip(rngs.iter()) {
            let si = &s[rng.clone()];
            let zi = &z[rng.clone()];
            cone.update_scaling(si, zi);
        }
    }

    fn set_identity_scaling(&mut self) {
        for cone in self.iter_mut() {
            cone.set_identity_scaling();
        }
    }

    #[allow(non_snake_case)]
    fn get_WtW_block(&self, WtWblock: &mut [T]) {
        for (cone,rng) in self.iter().zip(self.rng_blocks.iter()) {
            cone.get_WtW_block(&mut WtWblock[rng.clone()]);
        }
    }

    fn λ_circ_λ(&self, x: &mut [T]) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.λ_circ_λ(&mut x[rng.clone()]);
        }
    }

    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &mut x[rng.clone()];
            let yi = &y[rng.clone()];
            let zi = &z[rng.clone()];
            cone.circ_op(xi, yi, zi);
        }
    }

    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &mut x[rng.clone()];
            let zi = &z[rng.clone()];
            cone.λ_inv_circ_op(xi,zi);
        }
    }

    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &mut x[rng.clone()];
            let yi = &y[rng.clone()];
            let zi = &z[rng.clone()];
            cone.inv_circ_op(xi,yi,zi);
        }
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.shift_to_cone(&mut z[rng.clone()]);
        }
    }

    #[allow(non_snake_case)]
    fn gemv_W(
        &self,
        is_transpose: MatrixShape,
        x: &[T],
        y: &mut [T],
        α: T,
        β: T,
    ) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &x[rng.clone()];
            let yi = &mut y[rng.clone()];
            cone.gemv_W(is_transpose, xi, yi, α, β);
        }
    }

    #[allow(non_snake_case)]
    fn gemv_Winv(
        &self,
        is_transpose: MatrixShape,
        x: &[T],
        y: &mut [T],
        α: T,
        β: T,
    ) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &x[rng.clone()];
            let yi = &mut y[rng.clone()];
            cone.gemv_Winv(is_transpose, xi, yi, α, β);
        }
    }

    fn add_scaled_e(&self, x: &mut [T], α: T) {
        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.add_scaled_e(&mut x[rng.clone()], α);
        }
    }

    fn step_length(
        &self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
    ) -> (T, T) {
        let huge = T::recip(T::epsilon());
        let (mut αz, mut αs) = (huge, huge);

        for (cone,rng) in self.iter().zip(self.rng_cones.iter()) {
            let dzi = &dz[rng.clone()];
            let dsi = &ds[rng.clone()];
            let zi  = &z[rng.clone()];
            let si  = &s[rng.clone()];
            let (nextαz, nextαs) = cone.step_length(dzi, dsi, zi, si);
            αz = T::min(αz, nextαz);
            αs = T::min(αs, nextαs);
        }
        (αz, αs)
    }
}
