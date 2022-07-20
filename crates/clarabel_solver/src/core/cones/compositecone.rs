use super::*;
use clarabel_algebra::*;
use std::any::Any;
use std::collections::HashMap;
use std::ops::Range;

// ---------------------------------------------------
// We define some machinery here for enumerating the
// different cone types that can live in the composite cone
// ---------------------------------------------------

use core::hash::{Hash, Hasher};
use std::{cmp::PartialEq, mem::discriminant};

#[derive(Debug, Clone, Copy)]
pub enum SupportedCones<T> {
    ZeroConeT(usize),        // params: cone_dim
    NonnegativeConeT(usize), // params: cone_dim
    SecondOrderConeT(usize), // params: cone_dim
    PlaceHolderT(usize, T),  // params: cone_dim, exponent
}

impl<T> SupportedCones<T> {
    pub fn variant_name(&self) -> &'static str {
        match self {
            SupportedCones::ZeroConeT(_) => "ZeroConeT",
            SupportedCones::NonnegativeConeT(_) => "NonnegativeConeT",
            SupportedCones::SecondOrderConeT(_) => "SecondOrderConeT",
            SupportedCones::PlaceHolderT(_, _) => "PlaceHolderConeT",
        }
    }
}

impl<T: FloatT> std::fmt::Display for SupportedCones<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", &self.variant_name().to_string())
    }
}

// we will use the SupportedCones as a user facing marker
// for the constraint types, and then map them through
// a dictionary to get the internal cone representations.
// we will also make a HashMap of cone type counts, so need
// to define custom hashing and comparator ops
impl<T> Eq for SupportedCones<T> {}
impl<T> PartialEq for SupportedCones<T> {
    fn eq(&self, other: &Self) -> bool {
        discriminant(self) == discriminant(other)
    }
}

impl<T> Hash for SupportedCones<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);
    }
}

// -------------------------------------
// Here we make a type that will allow for convenient
// casting back to concrete cone types when needed.   This
// is used in particular for SOC, since the SOC can be
// represented an a sparse expanded format for LDL solvers.
// -------------------------------------

//AsAny is generic on T here, otherwise we get E0207 error
//complaining that our impl<T,U> AsAny has an unconstrained
//type parameter

pub trait AsAny<T> {
    fn as_any(&self) -> &dyn Any;
}

impl<T, U: Any + Cone<T>> AsAny<T> for U {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait AnyCone<T>: Cone<T> + AsAny<T> {}
impl<T, V: Cone<T> + AsAny<T>> AnyCone<T> for V {}
type BoxedCone<T> = Box<dyn AnyCone<T>>;

pub fn make_cone<T: FloatT>(cone: SupportedCones<T>) -> BoxedCone<T> {
    match cone {
        SupportedCones::NonnegativeConeT(dim) => Box::new(NonnegativeCone::<T>::new(dim)),
        SupportedCones::ZeroConeT(dim) => Box::new(ZeroCone::<T>::new(dim)),
        SupportedCones::SecondOrderConeT(dim) => Box::new(SecondOrderCone::<T>::new(dim)),
        SupportedCones::PlaceHolderT(_, _) => unimplemented!(),
    }
}

// -------------------------------------
// default composite cone type
// -------------------------------------

pub struct CompositeCone<T: FloatT = f64> {
    cones: Vec<BoxedCone<T>>,

    //Type tags and count of each cone
    pub types: Vec<SupportedCones<T>>,
    pub type_counts: HashMap<&'static str, usize>,

    //overall size of the composite cone
    numel: usize,
    degree: usize,

    //ranges for the indices of the constituent cones
    pub rng_cones: Vec<Range<usize>>,

    //ranges for the indices of the constituent WtW blocks
    //associated with each cone
    pub rng_blocks: Vec<Range<usize>>,
}

impl<T: FloatT> CompositeCone<T> {
    pub fn new(types: &[SupportedCones<T>]) -> Self {
        // make an internal copy to protect from user modification
        let types = types.to_vec();
        let ncones = types.len();
        let mut cones: Vec<BoxedCone<T>> = Vec::with_capacity(ncones);

        // create cones with the given dims
        for t in types.iter() {
            cones.push(make_cone(*t));
        }

        // Count the number of each cone type.
        // NB: ideally we could fix max capacity here,  but Enum::variant_count is not
        // yet a stable feature.  Capacity should be number of SupportedCones variants.
        // See: https://github.com/rust-lang/rust/issues/73662

        let mut type_counts = HashMap::new();
        for t in types.iter() {
            *type_counts.entry(&(*t.variant_name())).or_insert(0) += 1;
        }

        // count up elements and degree
        let numel = cones.iter().map(|c| c.numel()).sum();
        let degree = cones.iter().map(|c| c.degree()).sum();

        //ranges for the subvectors associated with each cone,
        //and the rangse for with the corresponding entries
        //in the WtW sparse block

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
            let stop = start + cone.numel();
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
                if cone.WtW_is_diagonal() {
                    nvars
                } else {
                    (nvars * (nvars + 1)) >> 1
                }
            };
            rngs.push(start..stop);
            start = stop;
        }
    }
    rngs
}

fn _make_headidx<T>(headidx: &mut [usize], cones: &[BoxedCone<T>])
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

impl<T> CompositeCone<T>
where
    T: FloatT,
{
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
    pub fn type_count(&self, cone_str: &'static str) -> usize {
        if self.type_counts.contains_key(cone_str) {
            self.type_counts[cone_str]
        } else {
            0
        }
    }
}

impl<T> Cone<T> for CompositeCone<T>
where
    T: FloatT,
{
    fn dim(&self) -> usize {
        panic!("dim() not well defined for the CompositeCone");
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
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
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
        let rngs = &self.rng_cones;

        for (cone, rng) in cones.iter_mut().zip(rngs.iter()) {
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
        for (cone, rng) in self.iter().zip(self.rng_blocks.iter()) {
            cone.get_WtW_block(&mut WtWblock[rng.clone()]);
        }
    }

    fn λ_circ_λ(&self, x: &mut [T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.λ_circ_λ(&mut x[rng.clone()]);
        }
    }

    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &mut x[rng.clone()];
            let yi = &y[rng.clone()];
            let zi = &z[rng.clone()];
            cone.circ_op(xi, yi, zi);
        }
    }

    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &mut x[rng.clone()];
            let zi = &z[rng.clone()];
            cone.λ_inv_circ_op(xi, zi);
        }
    }

    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &mut x[rng.clone()];
            let yi = &y[rng.clone()];
            let zi = &z[rng.clone()];
            cone.inv_circ_op(xi, yi, zi);
        }
    }

    fn shift_to_cone(&self, z: &mut [T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.shift_to_cone(&mut z[rng.clone()]);
        }
    }

    #[allow(non_snake_case)]
    fn gemv_W(&self, is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &x[rng.clone()];
            let yi = &mut y[rng.clone()];
            cone.gemv_W(is_transpose, xi, yi, α, β);
        }
    }

    #[allow(non_snake_case)]
    fn gemv_Winv(&self, is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let xi = &x[rng.clone()];
            let yi = &mut y[rng.clone()];
            cone.gemv_Winv(is_transpose, xi, yi, α, β);
        }
    }

    fn add_scaled_e(&self, x: &mut [T], α: T) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.add_scaled_e(&mut x[rng.clone()], α);
        }
    }

    fn step_length(&self, dz: &[T], ds: &[T], z: &[T], s: &[T]) -> (T, T) {
        let huge = T::max_value();
        let (mut αz, mut αs) = (huge, huge);

        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let dzi = &dz[rng.clone()];
            let dsi = &ds[rng.clone()];
            let zi = &z[rng.clone()];
            let si = &s[rng.clone()];
            let (nextαz, nextαs) = cone.step_length(dzi, dsi, zi, si);
            αz = T::min(αz, nextαz);
            αs = T::min(αs, nextαs);
        }
        (αz, αs)
    }
}
