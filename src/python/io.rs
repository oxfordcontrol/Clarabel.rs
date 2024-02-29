// Provides a Writer to allow for redirection of stdout and stderr streams
// to the ones configured for Python

use pyo3::ffi::{PyObject_CallMethod, PySys_GetObject, PySys_WriteStderr, PySys_WriteStdout};
use std::ffi::CString;
use std::os::raw::c_char;

macro_rules! make_python_stdio {
    ($name:ident, $pyfunc:ident, $modname:literal) => {
        pub(crate) struct $name {}
        impl std::io::Write for $name {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let cstr = std::ffi::CString::new(buf).unwrap();
                unsafe {
                    $pyfunc(cstr.as_ptr() as *const c_char);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                // call the python sys.stdout.flush()
                unsafe {
                    let stdout_str = CString::new($modname).unwrap();
                    let stdout_obj = PySys_GetObject(stdout_str.as_ptr() as *const c_char);
                    let flush_str = CString::new("flush").unwrap();
                    PyObject_CallMethod(
                        stdout_obj,
                        flush_str.as_ptr() as *const c_char,
                        std::ptr::null(),
                    );
                }
                Ok(())
            }
        }
    };
}
make_python_stdio!(PythonStdout, PySys_WriteStdout, "__stdout__");
make_python_stdio!(PythonStderr, PySys_WriteStderr, "__stderr__");

pub(crate) fn stdout() -> PythonStdout {
    PythonStdout {}
}

#[allow(dead_code)]
pub(crate) fn stderr() -> PythonStderr {
    PythonStderr {}
}
