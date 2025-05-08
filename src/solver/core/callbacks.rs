// ---------------------------------
// enum for managing callbacks
// ---------------------------------

pub(crate) type CallbackFcnFFI<FFI> = extern "C" fn(info: *const FFI) -> std::ffi::c_int;
pub trait ClarabelCallbackFn<I>: Fn(&I) -> bool + Send + Sync {}
impl<I, T: Fn(&I) -> bool + Send + Sync> ClarabelCallbackFn<I> for T {}

#[derive(Default)]
pub(crate) enum Callback<I, FFI> {
    #[default]
    None,
    Rust(Box<dyn ClarabelCallbackFn<I>>),
    C(CallbackFcnFFI<FFI>),
}

impl<I, FFI> std::fmt::Debug for Callback<I, FFI> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Callback::None => write!(f, "Callback::None"),
            Callback::Rust(_) => write!(f, "Callback::Rust(<closure>)"),
            Callback::C(fcn) => write!(f, "Callback::C({:?})", fcn),
        }
    }
}

impl<I, FFI> Callback<I, FFI>
where
    FFI: From<I>,
    I: Clone + Sized,
{
    // Call the callback function
    fn call(&self, info: &I) -> bool {
        match self {
            Callback::None => false,
            Callback::Rust(f) => f(info),
            Callback::C(f) => {
                let ffi_info = FFI::from(info.clone());
                f(&ffi_info as *const FFI) != (0 as std::ffi::c_int)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct SolverCallbacks<I, FFI> {
    /// callback for termination
    pub termination_callback: Callback<I, FFI>,
}

impl<I, FFI> Default for SolverCallbacks<I, FFI> {
    // Create a new set of callbacks
    fn default() -> Self {
        Self {
            termination_callback: Callback::None,
        }
    }
}

impl<I, FFI> SolverCallbacks<I, FFI>
where
    FFI: From<I>,
    I: Clone + Sized,
{
    pub(crate) fn check_termination(&self, info: &I) -> bool {
        // check termination conditions
        self.termination_callback.call(info)
    }
}
