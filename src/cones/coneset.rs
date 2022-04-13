use crate::algebra::*;
use std::collections::HashMap;
use super::*;

// -------------------------------------
// Cone Set (top level composite cone type)
// -------------------------------------

pub struct ConeSet<T: FloatT = f64> {

    cones: Vec<Box<dyn ConvexCone<T>>>,

    //Type tags and count of each cone
    types: Vec<SupportedCones>,
    type_counts : HashMap<SupportedCones,i32>,

    //overall size of the composite cone
    numel : usize,
    degree : usize,

    // a vector showing the overall index of the
    // first element in each cone.  For convenience
    headidx : Vec<usize>
}

impl<T: FloatT> ConeSet<T> {
    pub fn new(types: &[SupportedCones], dims : &[usize]) -> Self {

        assert_eq!(types.len(),dims.len());

        // make an internal copy to protect from user modification
        let types = types.clone().to_vec();
        let ncones = types.len();
        let mut cones : Vec<Box<dyn ConvexCone<T>>>  = Vec::with_capacity(ncones);

        // create cones with the given dims
        for (i,dim) in dims.iter().enumerate(){
            cones.push(cone_dict(types[i],*dim));
        }

        //  count the number of each cone type.
        // PJG: could perhaps fix max capacity here but Enum::variant_count is not
        // yet a stable feature.  Capacity should be number of SupportedCones variants

        let mut type_counts = HashMap::new();
        for t in types.iter(){
            *type_counts.entry(*t).or_insert(0) += 1;
        }

        // count up elements and degree
        let numel  = cones.iter().map(|c| c.numel()).sum();
        let degree = cones.iter().map(|c| c.degree()).sum();

        //headidx gives the index of the first element
        //of each constituent cone
        let mut headidx = vec![0; cones.len()];
        _coneset_make_headidx(&mut headidx,&cones);


        Self {
            cones : cones,
            types : types,
            type_counts : type_counts,
            numel : numel,
            degree : degree,
            headidx : headidx,
        }
    }
}

fn _coneset_make_headidx<T>(headidx: &mut [usize], cones: &[Box<dyn ConvexCone<T>>])
    where T: FloatT
{

    if cones.len() > 0 {
         // index of first element in each cone
        headidx[0] = 1;
        for i in 2..headidx.len() {
            headidx[i] = headidx[i-1] + cones[i-1].numel();
        }
    }
}


// ConeSet traits mirror the ConvexCone traits, but
// we take here arguments of type ConicVector everywhere.
// PJG: Could probably be done via generic instead of defining
// everything again here.
// PJG: This is a big mess since now I also have to make all
// the consituent functions pub
impl<T: FloatT> ConeSet<T> {

    fn degree(&self) -> usize {self.degree}
    pub fn numel(&self) -> usize {self.numel}

    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {

        //let mut any_changed = false;

        // we will update e <- δ .* e using return values
        // from this function.  default is to do nothing at all
        δ.fill(T::one());
        true //PJG fix me
    }

    fn update_scaling(&mut self, s: &[T], z: &[T]) {}
    fn set_identity_scaling(&mut self) {}
    #[allow(non_snake_case)]
    fn get_WtW_block(&self, WtWblock: &mut [T]) {}
    fn λ_circ_λ(&self, x: &mut [T]) {}
    fn circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {}
    fn λ_inv_circ_op(&self, x: &mut [T], z: &[T]) {}
    fn inv_circ_op(&self, x: &mut [T], y: &[T], z: &[T]) {}
    fn shift_to_cone(&self, z: &mut [T]) {}
    #[allow(non_snake_case)]
    fn gemv_W(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {}
    #[allow(non_snake_case)]
    fn gemv_Winv(&self, _is_transpose: MatrixShape, x: &[T], y: &mut [T], α: T, β: T) {}
    fn add_scaled_e(&self, x: &mut [T], α: T) {}
    fn step_length(&self, dz: &[T], ds: &[T], z: &[T], s: &[T]) -> (T, T)
    {
        (T::zero(),T::zero())
    }
}


impl<T: FloatT> ConeSet<T> {

    pub fn len(&self) -> usize {self.cones.len()}
    pub fn iter(&self) -> std::slice::Iter<'_, Box<dyn ConvexCone<T>>> {self.cones.iter()}

}
