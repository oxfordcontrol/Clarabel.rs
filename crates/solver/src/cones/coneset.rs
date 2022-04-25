use super::*;
use crate::algebra::*;
use crate::conicvector::ConicVector;
use std::collections::HashMap;
use std::any::Any;

// -------------------------------------
// Cone Set (top level composite cone type)
// -------------------------------------

type BoxedCone<T> = Box<dyn Cone<T, [T], [T]>>;

pub struct ConeSet<T: FloatT = f64> {

    cones: Vec<BoxedCone<T>>,

    //Type tags and count of each cone
    pub types: Vec<SupportedCones>,
    pub type_counts: HashMap<SupportedCones, usize>,

    //overall size of the composite cone
    numel: usize,
    degree: usize,

    // a vector showing the overall index of the
    // first element in each cone.  For convenience
    pub headidx: Vec<usize>,
}

impl<T: FloatT> ConeSet<T> {
    pub fn new(types: &[SupportedCones], dims: &[usize]) -> Self {
        assert_eq!(types.len(), dims.len());

        // make an internal copy to protect from user modification
        let types = types.to_vec();
        let ncones = types.len();
        let mut cones: Vec<BoxedCone<T>> = Vec::with_capacity(ncones);

        // create cones with the given dims
        for (i, dim) in dims.iter().enumerate() {
            cones.push(cone_dict(types[i], *dim));
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

        //headidx gives the index of the first element
        //of each constituent cone
        let mut headidx = vec![0; cones.len()];
        _coneset_make_headidx(&mut headidx, &cones);

        Self {
            cones, types, type_counts, numel, degree, headidx,
        }
    }
}

fn _coneset_make_headidx<T>(headidx: &mut [usize], cones: &[BoxedCone<T>])
where
    T: FloatT,
{
    if !cones.is_empty() {
        // index of first element in each cone
        headidx[0] = 1;
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
    pub fn anyref_by_idx(&self, idx: usize) -> &(dyn Any + '_)  {
        &self.cones[idx] as &(dyn Any + '_)
    }
}

impl<T> Cone<T, ConicVector<T>, Vec<Vec<T>>> for ConeSet<T>
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

    fn rectify_equilibration(&self, δ: &mut ConicVector<T>, e: &ConicVector<T>) -> bool {
        let mut any_changed = false;

        // we will update e <- δ .* e using return values
        // from this function.  default is to do nothing at all
        δ.fill(T::one());
        for (i, cone) in self.iter().enumerate() {
            any_changed |= cone.rectify_equilibration(δ.view_mut(i), e.view(i));
        }
        any_changed
    }

    fn WtW_is_diagonal(&self) -> bool{
        //This function should probably never be called since
        //we only us it to interrogate the blocks, but we can
        //implement something reasonable anyway
        let mut is_diag = true;
        for cone in self.iter(){
            is_diag &= cone.WtW_is_diagonal();
            if !is_diag {break;}
        }
        is_diag
    }

    fn update_scaling(&mut self, s: &ConicVector<T>, z: &ConicVector<T>) {
        for (i, cone) in self.iter_mut().enumerate() {
            cone.update_scaling(s.view(i), z.view(i));
        }
    }

    fn set_identity_scaling(&mut self) {
        for cone in self.iter_mut() {
            cone.set_identity_scaling();
        }
    }

    #[allow(non_snake_case)]
    fn get_WtW_block(&self, WtWblock: &mut Vec<Vec<T>>) {
        for (i, cone) in self.iter().enumerate() {
            cone.get_WtW_block(&mut WtWblock[i]);
        }
    }

    fn λ_circ_λ(&self, x: &mut ConicVector<T>) {
        for (i, cone) in self.iter().enumerate() {
            cone.λ_circ_λ(x.view_mut(i));
        }
    }

    fn circ_op(&self, x: &mut ConicVector<T>, y: &ConicVector<T>, z: &ConicVector<T>) {
        for (i, cone) in self.iter().enumerate() {
            cone.circ_op(x.view_mut(i), y.view(i), z.view(i));
        }
    }

    fn λ_inv_circ_op(&self, x: &mut ConicVector<T>, z: &ConicVector<T>) {
        for (i, cone) in self.iter().enumerate() {
            cone.λ_inv_circ_op(x.view_mut(i), z.view(i));
        }
    }

    fn inv_circ_op(&self, x: &mut ConicVector<T>, y: &ConicVector<T>, z: &ConicVector<T>) {
        for (i, cone) in self.iter().enumerate() {
            cone.inv_circ_op(x.view_mut(i), y.view(i), z.view(i));
        }
    }

    fn shift_to_cone(&self, z: &mut ConicVector<T>) {
        for (i, cone) in self.iter().enumerate() {
            cone.shift_to_cone(z.view_mut(i));
        }
    }

    #[allow(non_snake_case)]
    fn gemv_W(
        &self,
        is_transpose: MatrixShape,
        x: &ConicVector<T>,
        y: &mut ConicVector<T>,
        α: T,
        β: T,
    ) {
        for (i, cone) in self.iter().enumerate() {
            cone.gemv_W(is_transpose, x.view(i), y.view_mut(i), α, β);
        }
    }

    #[allow(non_snake_case)]
    fn gemv_Winv(
        &self,
        is_transpose: MatrixShape,
        x: &ConicVector<T>,
        y: &mut ConicVector<T>,
        α: T,
        β: T,
    ) {
        for (i, cone) in self.iter().enumerate() {
            cone.gemv_Winv(is_transpose, x.view(i), y.view_mut(i), α, β);
        }
    }

    fn add_scaled_e(&self, x: &mut ConicVector<T>, α: T) {
        for (i, cone) in self.iter().enumerate() {
            cone.add_scaled_e(x.view_mut(i), α);
        }
    }

    fn step_length(
        &self,
        dz: &ConicVector<T>,
        ds: &ConicVector<T>,
        z: &ConicVector<T>,
        s: &ConicVector<T>,
    ) -> (T, T) {
        let huge = T::recip(T::epsilon());
        let (mut αz, mut αs) = (huge, huge);

        for (i, cone) in self.iter().enumerate() {
            let (nextαz, nextαs) = cone.step_length(dz.view(i), ds.view(i), z.view(i), s.view(i));
            αz = T::min(αz, nextαz);
            αs = T::min(αs, nextαs);
        }
        (αz, αs)
    }
}
