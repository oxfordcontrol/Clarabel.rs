use super::*;
use crate::algebra::triangular_number;
use crate::solver::CoreSettings;
use std::collections::HashMap;
use std::iter::zip;
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
            let cone = make_cone(t);

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
                    triangular_number(nvars)
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
    pub(crate) fn get_type_count(&self, tag: SupportedConeTag) -> usize {
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
    fn degree(&self) -> usize {
        self.degree
    }

    fn numel(&self) -> usize {
        self.numel
    }

    fn is_symmetric(&self) -> bool {
        self._is_symmetric
    }

    fn is_sparse_expandable(&self) -> bool {
        //This should probably never be called
        //self.cones.iter().any(|cone| cone.is_sparse_expandable())
        unreachable!();
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        self.cones
            .iter()
            .all(|cone| cone.allows_primal_dual_scaling())
    }

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        let mut any_changed = false;

        // we will update e <- δ .* e using return values
        // from this function.  default is to do nothing at all
        δ.fill(T::one());
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
            let δi = &mut δ[rng.clone()];
            let ei = &e[rng.clone()];
            any_changed |= cone.rectify_equilibration(δi, ei);
        }
        any_changed
    }

    fn margins(&mut self, z: &mut [T], pd: PrimalOrDualCone) -> (T, T) {
        let mut α = T::max_value();
        let mut β = T::zero();
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let (αi, βi) = cone.margins(&mut z[rng.clone()], pd);
            α = T::min(α, αi);
            β += βi;
        }
        (α, β)
    }

    fn scaled_unit_shift(&self, z: &mut [T], α: T, pd: PrimalOrDualCone) {
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
            cone.scaled_unit_shift(&mut z[rng.clone()], α, pd);
        }
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
            cone.unit_initialization(&mut z[rng.clone()], &mut s[rng.clone()]);
        }
    }

    fn set_identity_scaling(&mut self) {
        for cone in self.iter_mut() {
            cone.set_identity_scaling();
        }
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        let mut is_scaling_success;
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let si = &s[rng.clone()];
            let zi = &z[rng.clone()];
            is_scaling_success = cone.update_scaling(si, zi, μ, scaling_strategy);
            if !is_scaling_success {
                return false;
            }
        }
        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        //This function should probably never be called since
        //we only us it to interrogate the blocks, but we can
        //implement something reasonable anyway
        self.cones.iter().all(|cone| cone.Hs_is_diagonal())
    }

    #[allow(non_snake_case)]
    fn get_Hs(&self, Hsblock: &mut [T]) {
        for (cone, rng) in zip(&self.cones, &self.rng_blocks) {
            cone.get_Hs(&mut Hsblock[rng.clone()]);
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], work: &mut [T]) {
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            cone.mul_Hs(&mut y[rng.clone()], &x[rng.clone()], &mut work[rng.clone()]);
        }
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        for (cone, rng) in zip(&self.cones, &self.rng_cones) {
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

        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let shifti = &mut shift[rng.clone()];
            let step_zi = &mut step_z[rng.clone()];
            let step_si = &mut step_s[rng.clone()];
            cone.combined_ds_shift(shifti, step_zi, step_si, σμ);
        }
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], z: &[T]) {
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let outi = &mut out[rng.clone()];
            let dsi = &ds[rng.clone()];
            let worki = &mut work[rng.clone()];
            let zi = &z[rng.clone()];
            cone.Δs_from_Δz_offset(outi, dsi, worki, zi);
        }
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let mut α = αmax;

        // Force symmetric cones first.
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
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

        if !self.is_symmetric() {
            α = T::min(settings.max_step_fraction, α);
        }
        // Force asymmetric cones last.
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
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

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();
        for (cone, rng) in zip(&mut self.cones, &self.rng_cones) {
            let zi = &z[rng.clone()];
            let si = &s[rng.clone()];
            let dzi = &dz[rng.clone()];
            let dsi = &ds[rng.clone()];
            barrier += cone.compute_barrier(zi, si, dzi, dsi, α);
        }
        barrier
    }
}
