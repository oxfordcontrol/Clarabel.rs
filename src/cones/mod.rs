// -------------------------------------
// Zero Cone
// -------------------------------------

pub struct ZeroCone {
    dim: usize,
}

impl ZeroCone {

    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim,
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn degree(&self) -> usize {
        0
    }

    pub fn numel(&self) -> usize {
        self.dim()
    }

    pub fn rectify_equilibration(&self, δ: &mut [f64], e: &[f64]) -> bool {
        δ.copy_from_slice(e);

        false
    }

    pub fn update_scaling(&self, _s: &[f64], _z: &[f64]) {
        //nothing to do
    }

    pub fn set_identity_scaling(&self) {
        //nothing to do
    }

    #[allow(non_snake_case)]
    pub fn get_WtW_block(&self, WtWblock: &mut [f64]) {
        WtWblock.fill(0.);
    }

    pub fn λ_circ_λ(&self, x: &mut [f64]) {
        x.fill(0.);
    }

    pub fn circ_op(&self, x: &mut [f64], _y: &[f64], _z: &[f64]) {
        x.fill(0.);
    }

    pub fn λ_inv_circ_op(&self, x: &mut [f64], _z: &[f64]) {
        x.fill(0.);
    }

    pub fn inv_circ_op(&self, x: &mut [f64]) {
        x.fill(0.);
    }

    pub fn shift_to_cone(&self, x: &mut [f64]) {
        x.fill(0.);
    }

    #[allow(non_snake_case)]
    pub fn gemv_W(&self, _is_transpose: bool, _x: &[f64], _y: &mut [f64], _α : f64, _β :  f64){
    }

}
//
//
// function rectify_equilibration!(
//     K::ZeroCone{T},
//     δ::AbstractVector{T},
//     e::AbstractVector{T}
// ) where{T}
//
//     #allow elementwise equilibration scaling
//     δ .= e
//     return false
// end
//
//
// function update_scaling!(
//     K::ZeroCone{T},
//     s::AbstractVector{T},
//     z::AbstractVector{T}
// ) where {T}
//
//     #nothing to do.
//     #This cone acts like λ = 0 everywhere.
//     return nothing
// end
//
// function set_identity_scaling!(
//     K::ZeroCone{T}
// ) where {T}
//
//     #do nothing.   "Identity" scaling will be zero for equalities
//     return nothing
// end
//
//
// function get_WtW_block!(
//     K::ZeroCone{T},
//     WtWblock::AbstractVector{T}
// ) where {T}
//
//     #expecting only a diagonal here, and
//     #setting it to zero since this is an
//     #equality condition
//     WtWblock .= zero(T)
//
//     return nothing
// end
//
// function λ_circ_λ!(
//     K::ZeroCone{T},
//     x::AbstractVector{T}
// ) where {T}
//
//     x .= zero(T)
//
// end
//
// # implements x = y ∘ z for the zero cone
// function circ_op!(
//     K::ZeroCone{T},
//     x::AbstractVector{T},
//     y::AbstractVector{T},
//     z::AbstractVector{T}
// ) where {T}
//
//     x .= zero(T)
//
//     return nothing
// end
//
// # implements x = λ \ z for the zerocone.
// # We treat λ as zero always for this cone
// function λ_inv_circ_op!(
//     K::ZeroCone{T},
//     x::AbstractVector{T},
//     z::AbstractVector{T}
// ) where {T}
//
//     x .= zero(T)
//
//     return nothing
// end
//
// # implements x = y \ z for the zero cone
// function inv_circ_op!(
//     K::ZeroCone{T},
//     x::AbstractVector{T},
//     y::AbstractVector{T},
//     z::AbstractVector{T}
// ) where {T}
//
//     x .= zero(T)
//
//     return nothing
// end
//
// # place vector into zero cone
// function shift_to_cone!(
//     K::ZeroCone{T},z::AbstractVector{T}
// ) where{T}
//
//     z .= zero(T)
//
//     return nothing
// end
//
// # implements y = αWx + βy for the zero cone
// function gemv_W!(
//     K::ZeroCone{T},
//     is_transpose::Symbol,
//     x::AbstractVector{T},
//     y::AbstractVector{T},
//     α::T,
//     β::T
// ) where {T}
//
//     #treat W like zero
//     y .= β.*y
//
//     return nothing
// end
//
// # implements y = αWx + βy for the nn cone
// function gemv_Winv!(
//     K::ZeroCone{T},
//     is_transpose::Symbol,
//     x::AbstractVector{T},
//     y::AbstractVector{T},
//     α::T,
//     β::T
// ) where {T}
//
//   #treat Winv like zero
//   y .= β.*y
//
//   return nothing
// end
//
// # implements y = y + αe for the zero cone
// function add_scaled_e!(
//     K::ZeroCone{T},
//     x::AbstractVector{T},α::T
// ) where {T}
//
//     #e = 0, do nothing
//     return nothing
//
// end
//
//
// function step_length(
//      K::ZeroCone{T},
//     dz::AbstractVector{T},
//     ds::AbstractVector{T},
//      z::AbstractVector{T},
//      s::AbstractVector{T}
// ) where {T}
//
//     #equality constraints allow arbitrary step length
//     huge = inv(eps(T))
//     return (huge,huge)
// end
