use super::*;
use crate::{algebra::AsFloatT, solver::CoreSettings};
use std::collections::HashMap;
use std::ops::Range;

// -------------------------------------
// default composite cone type
// -------------------------------------

pub struct CompositeCone<T: FloatT = f64> {
    cones: Vec<SupportedCone<T>>,

    //Type count for each cone type
    pub(crate) type_counts: HashMap<SupportedConeTag, usize>,

    //overall size of the composite cone
    pub(crate) numel: usize,
    pub(crate) degree: usize,

    //ranges for the indices of the constituent cones
    pub(crate) rng_cones: Vec<Range<usize>>,

    //ranges for the indices of the constituent Hs blocks
    //associated with each cone
    pub(crate) rng_blocks: Vec<Range<usize>>,

    // the flag for symmetric cone check
    _is_symmetric: bool,
}

impl<T> CompositeCone<T>
where
    T: FloatT,
{
    pub fn new(types: &[SupportedConeT<T>]) -> Self {
        // make an internal copy to protect from user modification
        let types = types.to_vec();
        let ncones = types.len();
        let mut cones: Vec<SupportedCone<T>> = Vec::with_capacity(ncones);

        // Count for the number of each cone type, indexed by SupportedConeTag
        // NB: ideally we could fix max capacity here,  but Enum::variant_count is not
        // yet a stable feature.  Capacity should be number of SupportedCone variants.
        // See: https://github.com/rust-lang/rust/issues/73662
        let mut type_counts = HashMap::new();

        // assumed symmetric to start
        let mut _is_symmetric = true;

        // create cones with the given dims
        for t in types.iter() {
            //make a new cone
            let cone = make_cone(*t);

            //update global problem symmetry
            _is_symmetric = _is_symmetric && cone.is_symmetric();

            //increment type counts
            *type_counts.entry(cone.as_tag()).or_insert(0) += 1;

            cones.push(cone);
        }

        // count up elements and degree
        let numel = cones.iter().map(|c| c.numel()).sum();
        let degree = cones.iter().map(|c| c.degree()).sum();

        //ranges for the subvectors associated with each cone,
        //and the rangse for with the corresponding entries
        //in the Hs sparse block

        let rng_cones = _make_rng_cones(&cones);
        let rng_blocks = _make_rng_blocks(&cones);

        Self {
            cones,
            //types,
            type_counts,
            numel,
            degree,
            rng_cones,
            rng_blocks,
            _is_symmetric,
        }
    }
}

fn _make_rng_cones<T>(cones: &[SupportedCone<T>]) -> Vec<Range<usize>>
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

fn _make_rng_blocks<T>(cones: &[SupportedCone<T>]) -> Vec<Range<usize>>
where
    T: FloatT,
{
    let mut rngs = Vec::with_capacity(cones.len());

    if !cones.is_empty() {
        let mut start = 0;
        for cone in cones {
            let nvars = cone.numel();
            let stop = start + {
                if cone.Hs_is_diagonal() {
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

fn _make_headidx<T>(headidx: &mut [usize], cones: &[SupportedCone<T>])
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
    pub fn iter(&self) -> std::slice::Iter<'_, SupportedCone<T>> {
        self.cones.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, SupportedCone<T>> {
        self.cones.iter_mut()
    }
    pub(crate) fn type_count(&self, tag: SupportedConeTag) -> usize {
        if self.type_counts.contains_key(&tag) {
            self.type_counts[&tag]
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

    fn is_symmetric(&self) -> bool {
        self._is_symmetric
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

    fn shift_to_cone(&self, z: &mut [T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.shift_to_cone(&mut z[rng.clone()]);
        }
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.unit_initialization(&mut z[rng.clone()], &mut s[rng.clone()]);
        }
    }

    fn set_identity_scaling(&mut self) {
        for cone in self.iter_mut() {
            cone.set_identity_scaling();
        }
    }

    fn update_scaling(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy) {
        let cones = &mut self.cones;
        let rngs = &self.rng_cones;

        for (cone, rng) in cones.iter_mut().zip(rngs.iter()) {
            let si = &s[rng.clone()];
            let zi = &z[rng.clone()];
            cone.update_scaling(si, zi, μ, scaling_strategy);
        }
    }

    fn Hs_is_diagonal(&self) -> bool {
        //This function should probably never be called since
        //we only us it to interrogate the blocks, but we can
        //implement something reasonable anyway
        let mut is_diag = true;
        for cone in self.iter() {
            is_diag &= cone.Hs_is_diagonal();
            if !is_diag {
                break;
            }
        }
        is_diag
    }

    #[allow(non_snake_case)]
    fn get_Hs(&self, Hsblock: &mut [T]) {
        for (cone, rng) in self.iter().zip(self.rng_blocks.iter()) {
            cone.get_Hs(&mut Hsblock[rng.clone()]);
        }
    }

    fn mul_Hs(&self, y: &mut [T], x: &[T], work: &mut [T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            cone.mul_Hs(&mut y[rng.clone()], &x[rng.clone()], &mut work[rng.clone()]);
        }
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let dsi = &mut ds[rng.clone()];
            let si = &s[rng.clone()];
            cone.affine_ds(dsi, si);
        }
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        // Here we must first explicitly borrow the subvector
        // of cones, since trying to access it using self.iter_mut
        // causes a borrow conflict with ranges.

        // It is necessary for the function to mutate self since
        // nonsymmetric cones modify their internal state when
        // computing the ds_shift

        let cones = &mut self.cones;
        let rngs = &self.rng_cones;

        for (cone, rng) in cones.iter_mut().zip(rngs) {
            let shifti = &mut shift[rng.clone()];
            let step_zi = &mut step_z[rng.clone()];
            let step_si = &mut step_s[rng.clone()];
            cone.combined_ds_shift(shifti, step_zi, step_si, σμ);
        }
    }

    fn Δs_from_Δz_offset(&self, out: &mut [T], ds: &[T], work: &mut [T]) {
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let outi = &mut out[rng.clone()];
            let dsi = &ds[rng.clone()];
            let worki = &mut work[rng.clone()];
            cone.Δs_from_Δz_offset(outi, dsi, worki);
        }
    }

    fn step_length(
        &self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let mut α = αmax;

        // Force symmetric cones first.
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            if !cone.is_symmetric() {
                continue;
            }
            let (dzi, dsi) = (&dz[rng.clone()], &ds[rng.clone()]);
            let (zi, si) = (&z[rng.clone()], &s[rng.clone()]);
            let (nextαz, nextαs) = cone.step_length(dzi, dsi, zi, si, settings, α);
            α = T::min(α, T::min(nextαz, nextαs));
        }

        // if we have any nonsymmetric cones, then back off from full steps slightly
        // so that centrality checks and logarithms don't fail right at the boundaries
        // PJG: is this still necessary?

        if !self.is_symmetric() {
            let ceil: T = (0.99_f64).as_T();
            α = T::min(ceil, α);
        }
        // Force asymmetric cones last.
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            if cone.is_symmetric() {
                continue;
            }
            let (dzi, dsi) = (&dz[rng.clone()], &ds[rng.clone()]);
            let (zi, si) = (&z[rng.clone()], &s[rng.clone()]);
            let (nextαz, nextαs) = cone.step_length(dzi, dsi, zi, si, settings, α);
            α = T::min(α, T::min(nextαz, nextαs));
        }

        (α, α)
    }

    fn compute_barrier(&self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();
        for (cone, rng) in self.iter().zip(self.rng_cones.iter()) {
            let zi = &z[rng.clone()];
            let si = &s[rng.clone()];
            let dzi = &dz[rng.clone()];
            let dsi = &ds[rng.clone()];
            barrier += cone.compute_barrier(zi, si, dzi, dsi, α);
        }
        barrier
    }
}
