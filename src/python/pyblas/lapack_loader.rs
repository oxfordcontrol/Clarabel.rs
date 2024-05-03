use super::{lapack_types::*, *};

impl PyLapackPointers {
    pub(crate) fn new(py: Python) -> PyResult<PyLapackPointers> {
        let api = get_pyx_capi(py, "scipy.linalg.cython_lapack")?;

        unsafe {
            let ptrs = PyLapackPointers {
                dsyevr_: get_ptr!(api, "dsyevr"),
                ssyevr_: get_ptr!(api, "ssyevr"),
                dpotrf_: get_ptr!(api, "dpotrf"),
                spotrf_: get_ptr!(api, "spotrf"),
                dpotrs_: get_ptr!(api, "dpotrs"),
                spotrs_: get_ptr!(api, "spotrs"),
                dgesdd_: get_ptr!(api, "dgesdd"),
                sgesdd_: get_ptr!(api, "sgesdd"),
                dgesvd_: get_ptr!(api, "dgesvd"),
                sgesvd_: get_ptr!(api, "sgesvd"),
                dgesv_: get_ptr!(api, "dgesv"),
                sgesv_: get_ptr!(api, "sgesv"),
            };
            Ok(ptrs)
        }
    }
}
