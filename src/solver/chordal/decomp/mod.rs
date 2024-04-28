#![allow(non_snake_case)]

mod augment_compact;
mod augment_standard;
mod psd_completion;
mod reverse_compact;
mod reverse_standard;

use super::chordal_info::ChordalInfo;
use crate::{
    algebra::*,
    solver::{DefaultSettings, DefaultVariables, SupportedConeT},
};

impl<T> ChordalInfo<T>
where
    T: FloatT,
{
    #[allow(clippy::type_complexity)]
    pub(crate) fn decomp_augment(
        &mut self,
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        settings: &DefaultSettings<T>,
    ) -> (
        CscMatrix<T>,
        Vec<T>,
        CscMatrix<T>,
        Vec<T>,
        Vec<SupportedConeT<T>>,
    ) {
        if settings.chordal_decomposition_compact {
            self.decomp_augment_compact(P, q, A, b)
        } else {
            self.decomp_augment_standard(P, q, A, b)
        }
    }

    pub(crate) fn decomp_reverse(
        &self,
        old_vars: &DefaultVariables<T>,
        old_cones: &[SupportedConeT<T>],
        settings: &DefaultSettings<T>,
    ) -> DefaultVariables<T>
    where
        T: FloatT,
    {
        // We should have either H (for standard decomp) or cone_maps (for compact decomp)
        // but never both, and they should be consistent with the settings
        assert_eq!(settings.chordal_decomposition_compact, self.H.is_none());
        assert_eq!(
            settings.chordal_decomposition_compact,
            self.cone_maps.is_some()
        );

        // here `old_cones' are the ones that were used internally
        // in the solver, producing internal solution in `old_vars'
        // the cones for the problem as provided by the user or the
        // upstream presolver are held internally in chordal_info.cones

        let (n, m) = self.init_dims;
        let mut new_vars = DefaultVariables::<T>::new(n, m);

        new_vars.x.copy_from(&old_vars.x[0..n]);

        // reassemble the original variables s and z
        if settings.chordal_decomposition_compact {
            self.decomp_reverse_compact(&mut new_vars, old_vars, old_cones);
        } else {
            self.decomp_reverse_standard(&mut new_vars, old_vars, old_cones);
        }

        if settings.chordal_decomposition_complete_dual {
            self.psd_completion(&mut new_vars);
        }

        new_vars
    }
}
