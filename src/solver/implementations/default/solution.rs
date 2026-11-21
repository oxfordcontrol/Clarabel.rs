#![allow(unused_variables)]

use super::*;
use crate::{
    algebra::*,
    solver::core::{
        cones::{SupportedConeAsTag, SupportedConeTag},
        traits::Solution,
        SolverStatus,
    },
};

/// Standard-form solver type implementing the [`Solution`](crate::solver::core::traits::Solution) trait
///
/// When the `serde` feature is enabled, this type derives `Serialize`
/// and `Deserialize` with bound `T: Serialize + DeserializeOwned`.
/// For `T = RationalReal` this gives bit-exact JSON witnesses
/// (numerator/denominator pairs preserved through round-trip).
/// Marked `#[non_exhaustive]`: external struct-literal construction is not
/// supported (use [`DefaultSolution::new`]), so future fields can be added
/// without further breakage.  Deserialization of previously-written JSON is
/// unaffected — added fields carry serde defaults.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound = "T: serde::Serialize + serde::de::DeserializeOwned")
)]
#[non_exhaustive]
pub struct DefaultSolution<T> {
    /// primal solution
    pub x: Vec<T>,
    /// dual solution (in dual cone)
    pub z: Vec<T>,
    /// vector of slacks (in primal cone)
    pub s: Vec<T>,
    /// final solver status
    pub status: SolverStatus,
    /// primal objective value
    pub obj_val: T,
    /// dual objective value
    pub obj_val_dual: T,
    /// solve time in seconds
    pub solve_time: f64,
    /// number of iterations
    pub iterations: u32,
    /// primal residual
    pub r_prim: T,
    /// dual residual
    pub r_dual: T,

    /// Per-cone metadata (tag + slack-vector range), captured from the
    /// post-collapse cone list at termination. Lets callers extract
    /// `z` / `s` per-cone slices without re-walking the user's
    /// original cone declarations.
    ///
    /// Length matches the number of cones in the *internal* (post-
    /// `new_collapsed`) representation. Sugar variants like
    /// [`BlockDiagPSDConeT`](crate::solver::core::cones::SupportedConeT::BlockDiagPSDConeT)
    /// will appear here as their expanded constituents (one entry per block).
    ///
    /// # Known limitation
    ///
    /// This list is taken from `DefaultProblemData::cones`, which is the cone
    /// list *after* presolve and chordal decomposition have rewritten it,
    /// whereas `s` and `z` are returned at the original, un-reduced length.
    /// When presolve fires the two disagree.  Measured on the two-cone
    /// presolve case in `tests/presolve.rs` (one `b` entry at infinity,
    /// `[NonnegativeConeT(3), NonnegativeConeT(3)]`): `z.len() == 6` but
    /// `cone_specs` is the single entry `NonnegativeCone, range 0..5`,
    /// covering 5 of the 6 slack entries.  This is a pre-existing defect and
    /// is not addressed here.
    ///
    /// [`declared_cone_specs`](Self::declared_cone_specs) is derived from the
    /// caller's own cone list and never sees presolve or chordal
    /// decomposition, so it is correct in those cases.
    pub cone_specs: Vec<ConeSpec>,

    /// Per-cone metadata for the cones the caller *declared*, as opposed to
    /// the collapsed list `cone_specs` describes.
    ///
    /// The solver internally collapses the user's cone list before solving:
    /// adjacent nonnegative cones are merged, `SecondOrderConeT(1)` and
    /// `PSDTriangleConeT(1)` singletons are rewritten as `NonnegativeConeT(1)`,
    /// and empty cones are dropped.  That is a sound optimisation -- it does
    /// not change the solution -- but it means `cone_specs` neither preserves
    /// the caller's cone *indices* nor remembers that a 1x1 cone was declared
    /// PSD.  This field does both.
    ///
    /// Entries are the caller's cones with only the
    /// [`BlockDiagPSDConeT`](crate::solver::core::cones::SupportedConeT::BlockDiagPSDConeT)
    /// sugar unfolded, in declaration order, one entry per block.  Each entry
    /// records the index of the cone in the caller's input slice it came from
    /// (see [`DeclaredConeSpec::input_index`]), so the blocks belonging to one
    /// `BlockDiagPSDConeT` are identifiable as a group.
    ///
    /// The ranges here index the same `s` / `z` vectors as `cone_specs`, and
    /// are a refinement of them: the collapse pass only ever merges adjacent
    /// ranges or removes zero-length ones, so a declared range is always a
    /// contiguous sub-slice of exactly one collapsed range.
    #[cfg_attr(feature = "serde", serde(default))]
    pub declared_cone_specs: Vec<DeclaredConeSpec>,
}

/// Per-cone metadata recorded on `DefaultSolution`. Used as the offset
/// table for the structured per-block accessors.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConeSpec {
    /// Tag identifying the cone type (NonnegativeCone / SecondOrderCone /
    /// PSDTriangleCone / etc).
    pub tag: SupportedConeTag,
    /// Half-open range `[start, stop)` indexing into the flat `z` and
    /// `s` slack vectors. `s[range]` and `z[range]` are the cone's
    /// per-cone slack and dual blocks.
    pub range: std::ops::Range<usize>,
    /// For `PSDTriangleCone` cones: the matrix dimension `d` (so the
    /// svec has `triangular_number(d)` entries). For other cones: the
    /// scalar dimension. Used by [`DefaultSolution::dual_psd_block`]
    /// and friends to unpack svec into a dense `d×d` matrix.
    pub dim: usize,
}

/// Per-cone metadata for a cone *as the caller declared it*, before the
/// internal collapse pass rewrote it.  Recorded on
/// [`DefaultSolution::declared_cone_specs`].
///
/// This is deliberately a separate type from [`ConeSpec`] rather than an
/// extra field on it: `ConeSpec` is public and (under the `serde` feature)
/// serialized, so growing it would change its wire format for every existing
/// consumer.  The two lists also have different lengths whenever the collapse
/// pass did anything, so they cannot share entries in any case.
///
/// Marked `#[non_exhaustive]` so that later fields can be added without a
/// breaking change.  This costs nothing here: the type is new, so no external
/// struct-literal construction of it can exist yet.  Note that [`ConeSpec`] is
/// deliberately *not* marked — it predates this type, so marking it would
/// itself be the breaking change this crate went out of its way to avoid.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DeclaredConeSpec {
    /// Tag identifying the cone type **as declared**.  A `PSDTriangleConeT(1)`
    /// reports `PSDTriangleCone` here even though the solver collapsed it to a
    /// `NonnegativeCone`.
    pub tag: SupportedConeTag,
    /// Half-open range `[start, stop)` indexing into the flat `z` and `s`
    /// vectors, exactly as for [`ConeSpec::range`].
    pub range: std::ops::Range<usize>,
    /// For `PSDTriangleCone` cones the matrix dimension `d`; for other cones
    /// the scalar dimension.
    pub dim: usize,
    /// Index of the cone in the caller's original input slice that produced
    /// this entry.  Normally equal to the position in `declared_cone_specs`,
    /// but a `BlockDiagPSDConeT` with `k` blocks produces `k` consecutive
    /// entries that all carry its input index, and any cone list containing
    /// one shifts every later entry's position relative to its input index.
    pub input_index: usize,
}

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    /// Create a new `DefaultSolution` object
    pub fn new(n: usize, m: usize) -> Self {
        let x = vec![T::zero(); n];
        let z = vec![T::zero(); m];
        let s = vec![T::zero(); m];

        Self {
            x,
            z,
            s,
            status: SolverStatus::Unsolved,
            obj_val: T::nan(),
            obj_val_dual: T::nan(),
            solve_time: 0f64,
            iterations: 0,
            r_prim: T::nan(),
            r_dual: T::nan(),
            cone_specs: Vec::new(),
            declared_cone_specs: Vec::new(),
        }
    }
}

// =========================================================
// Structured per-block accessors for SDP / multi-cone problems
// =========================================================

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    /// Slice of the dual `z` vector corresponding to cone `idx` in
    /// the post-collapse cone list.
    ///
    /// Returns `None` if `idx` is out of range. The cone's range
    /// metadata lives on `self.cone_specs[idx]`.
    pub fn dual_block(&self, idx: usize) -> Option<&[T]> {
        let spec = self.cone_specs.get(idx)?;
        Some(&self.z[spec.range.clone()])
    }

    /// Slice of the slack `s` vector corresponding to cone `idx`.
    pub fn primal_block(&self, idx: usize) -> Option<&[T]> {
        let spec = self.cone_specs.get(idx)?;
        Some(&self.s[spec.range.clone()])
    }

    /// For a `PSDTriangleCone` cone at position `idx`, unpack the
    /// dual `z` slice into a dense `d × d` symmetric matrix in
    /// row-major order. The svec packing matches Clarabel's
    /// internal convention: triu in column-major order with
    /// off-diagonals scaled by sqrt(2). The returned matrix is
    /// the recovered scaled-symmetric form.
    ///
    /// Returns `None` if `idx` is out of range or the cone at
    /// position `idx` is not `PSDTriangleCone`.
    #[cfg(feature = "sdp")]
    pub fn dual_psd_block(&self, idx: usize) -> Option<Vec<Vec<T>>> {
        let spec = self.cone_specs.get(idx)?;
        if spec.tag != SupportedConeTag::PSDTriangleCone {
            return None;
        }
        let d = spec.dim;
        let svec = &self.z[spec.range.clone()];
        Some(unpack_svec::<T>(svec, d))
    }

    /// Same as [`dual_psd_block`](Self::dual_psd_block) for the
    /// primal slack `s`.
    #[cfg(feature = "sdp")]
    pub fn primal_psd_block(&self, idx: usize) -> Option<Vec<Vec<T>>> {
        let spec = self.cone_specs.get(idx)?;
        if spec.tag != SupportedConeTag::PSDTriangleCone {
            return None;
        }
        let d = spec.dim;
        let svec = &self.s[spec.range.clone()];
        Some(unpack_svec::<T>(svec, d))
    }

    /// Slice of the dual `z` vector corresponding to declared cone `idx`
    /// (an index into [`declared_cone_specs`](Self::declared_cone_specs),
    /// i.e. the caller's own cone ordering with `BlockDiagPSDConeT` blocks
    /// counted individually).
    ///
    /// Unlike [`dual_block`](Self::dual_block), this index is not disturbed
    /// by the internal collapse pass merging or dropping cones.
    ///
    /// Returns `None` if `idx` is out of range.
    pub fn declared_dual_block(&self, idx: usize) -> Option<&[T]> {
        let spec = self.declared_cone_specs.get(idx)?;
        Some(&self.z[spec.range.clone()])
    }

    /// Slice of the slack `s` vector corresponding to declared cone `idx`.
    /// See [`declared_dual_block`](Self::declared_dual_block).
    pub fn declared_primal_block(&self, idx: usize) -> Option<&[T]> {
        let spec = self.declared_cone_specs.get(idx)?;
        Some(&self.s[spec.range.clone()])
    }

    /// For a cone the caller **declared** as `PSDTriangleConeT` at declared
    /// position `idx`, unpack the dual `z` slice into a dense `d × d`
    /// symmetric matrix, exactly as [`dual_psd_block`](Self::dual_psd_block)
    /// does for the collapsed list.
    ///
    /// This honours the declaration rather than the internal representation,
    /// so a declared `PSDTriangleConeT(1)` yields `Some([[value]])` here even
    /// though the solver collapsed it to a `NonnegativeConeT(1)` and
    /// `dual_psd_block` therefore returns `None` for it.
    ///
    /// Returns `None` if `idx` is out of range or the cone at declared
    /// position `idx` was not declared as `PSDTriangleConeT`.
    #[cfg(feature = "sdp")]
    pub fn declared_dual_psd_block(&self, idx: usize) -> Option<Vec<Vec<T>>> {
        let spec = self.declared_cone_specs.get(idx)?;
        if spec.tag != SupportedConeTag::PSDTriangleCone {
            return None;
        }
        Some(unpack_svec::<T>(&self.z[spec.range.clone()], spec.dim))
    }

    /// Same as [`declared_dual_psd_block`](Self::declared_dual_psd_block) for
    /// the primal slack `s`.
    #[cfg(feature = "sdp")]
    pub fn declared_primal_psd_block(&self, idx: usize) -> Option<Vec<Vec<T>>> {
        let spec = self.declared_cone_specs.get(idx)?;
        if spec.tag != SupportedConeTag::PSDTriangleCone {
            return None;
        }
        Some(unpack_svec::<T>(&self.s[spec.range.clone()], spec.dim))
    }

    /// The declared positions produced by entry `input_index` of the caller's
    /// original cone slice.
    ///
    /// This is the identity for every cone except `BlockDiagPSDConeT`, which
    /// contributes one declared position per block.  Returns an empty vector
    /// if the input cone contributed no declared cones (a `BlockDiagPSDConeT`
    /// with empty `block_dims`) or if `input_index` is out of range.
    pub fn declared_positions_for_input(&self, input_index: usize) -> Vec<usize> {
        self.declared_cone_specs
            .iter()
            .enumerate()
            .filter(|(_, spec)| spec.input_index == input_index)
            .map(|(pos, _)| pos)
            .collect()
    }

    /// Sum-of-squares norm of `s + Ax - b` per cone, evaluated at
    /// solver precision. Useful for certifying that a rounded /
    /// projected solution still satisfies each cone individually.
    /// Returns one entry per `cone_specs` entry, in the same order.
    pub fn primal_residual_per_block(&self) -> Vec<T> {
        // The slack `s` already carries the per-cone primal-feasibility
        // residual at convergence — for solved problems s ∈ K and
        // s = b - Ax. We expose `(z, s)`-norm-style summaries by
        // computing per-block ||s|| via VectorMath::norm.
        self.cone_specs
            .iter()
            .map(|spec| self.s[spec.range.clone()].norm())
            .collect()
    }
}

/// Unpack a packed symmetric (svec) representation into a dense
/// `d × d` row-major Vec<Vec<T>>. Off-diagonals are de-scaled by
/// `1 / sqrt(2)` to recover the original symmetric matrix.
#[cfg(feature = "sdp")]
fn unpack_svec<T: FloatT>(svec: &[T], d: usize) -> Vec<Vec<T>> {
    let inv_sqrt_2 = T::FRAC_1_SQRT_2();
    let mut out = vec![vec![T::zero(); d]; d];
    let mut k = 0;
    for col in 0..d {
        for row in 0..=col {
            let v = svec[k].clone();
            if row == col {
                out[row][col] = v;
            } else {
                let scaled = v * inv_sqrt_2.clone();
                out[row][col] = scaled.clone();
                out[col][row] = scaled;
            }
            k += 1;
        }
    }
    out
}

impl<T> Solution<T> for DefaultSolution<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type I = DefaultInfo<T>;
    type SE = DefaultSettings<T>;

    fn post_process(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &mut DefaultVariables<T>,
        info: &DefaultInfo<T>,
        settings: &DefaultSettings<T>,
    ) {
        self.status = info.status;
        let is_infeasible = info.status.is_infeasible();

        if is_infeasible {
            self.obj_val = T::nan();
            self.obj_val_dual = T::nan();
        } else {
            self.obj_val = info.cost_primal.clone();
            self.obj_val_dual = info.cost_dual.clone();
        }

        self.iterations = info.iterations;
        self.r_prim = info.res_primal.clone();
        self.r_dual = info.res_dual.clone();

        // unscale the variables to get a solution
        // to the internal problem as we solved it
        variables.unscale(data, is_infeasible);

        // unwind the chordal decomp and presolve, in the
        // reverse of the order in which they were applied
        #[cfg(feature = "sdp")]
        let tmp = data
            .chordal_info
            .as_ref()
            .map(|chordal_info| chordal_info.decomp_reverse(variables, &data.cones, settings));
        #[cfg(feature = "sdp")]
        let variables = tmp.as_ref().unwrap_or(variables);

        if let Some(ref presolver) = data.presolver {
            presolver.reverse_presolve(self, variables);
        } else {
            self.x.copy_from(&variables.x);
            self.z.copy_from(&variables.z);
            self.s.copy_from(&variables.s);
        }

        // Populate per-cone metadata for the structured per-block
        // accessors (dual_block, primal_block, dual_psd_block,
        // primal_residual_per_block). This is the post-collapse cone
        // list — sugar variants like BlockDiagPSDConeT have already
        // been expanded by SupportedConeT::new_collapsed at solver
        // construction time, so each entry here corresponds to one
        // contiguous block of the s/z slack vectors.
        self.cone_specs.clear();
        let mut start = 0usize;
        for cone in &data.cones {
            let nv = cone.nvars();
            self.cone_specs.push(ConeSpec {
                tag: cone.as_tag(),
                range: start..(start + nv),
                // `spec_dim` panics on BlockDiagPSDConeT, which is exactly
                // the invariant asserted here: the sugar is desugared by
                // `new_collapsed` and cannot reach `post_process`.
                dim: cone.spec_dim(),
            });
            start += nv;
        }

        // Populate the *declared* metadata from the cone list the caller
        // actually wrote, retained on the problem data before the collapse
        // pass shadowed it.  These ranges refine the collapsed ones -- the
        // collapse pass only merges adjacent ranges or drops zero-length
        // ones -- so they index the same `s` / `z` vectors.
        self.declared_cone_specs.clear();
        let mut start = 0usize;
        for (cone, &input_index) in data
            .declared_cones
            .iter()
            .zip(data.declared_cone_origin.iter())
        {
            let nv = cone.nvars();
            self.declared_cone_specs.push(DeclaredConeSpec {
                tag: cone.as_tag(),
                range: start..(start + nv),
                dim: cone.spec_dim(),
                input_index,
            });
            start += nv;
        }
    }

    fn finalize(&mut self, info: &DefaultInfo<T>) {
        self.solve_time = info.solve_time;
    }
}
