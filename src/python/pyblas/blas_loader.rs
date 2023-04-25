use super::{blas_types::*, *};

impl PyBlasPointers {
    pub(crate) fn new(py: Python) -> PyResult<PyBlasPointers> {
        let api = get_pyx_capi(py, "scipy.linalg.cython_blas")?;

        unsafe {
            let ptrs = PyBlasPointers {
                ddot_: get_ptr!(api, "ddot"),
                sdot_: get_ptr!(api, "sdot"),
                dgemm_: get_ptr!(api, "dgemm"),
                sgemm_: get_ptr!(api, "sgemm"),
                dgemv_: get_ptr!(api, "dgemv"),
                sgemv_: get_ptr!(api, "sgemv"),
                dsymv_: get_ptr!(api, "dsymv"),
                ssymv_: get_ptr!(api, "ssymv"),
                dsyrk_: get_ptr!(api, "dsyrk"),
                ssyrk_: get_ptr!(api, "ssyrk"),
                dsyr2k_: get_ptr!(api, "dsyr2k"),
                ssyr2k_: get_ptr!(api, "ssyr2k"),
            };
            Ok(ptrs)
        }
    }
}
