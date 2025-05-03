/// trait for defining FFI data counterparts as associated types
#[allow(missing_docs)]
pub trait ClarabelFFI<I> {
    type FFI: From<I>;
}
