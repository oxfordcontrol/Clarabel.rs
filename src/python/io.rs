// Provides a Writer to allow for redirection of stdout and stderr streams
// to the ones configured for Python

use pyo3::ffi::{PySys_WriteStderr, PySys_WriteStdout};

macro_rules! make_python_stdio {
    ($name:ident, $pyfunc:ident) => {
        pub(crate) struct $name {}
        impl std::io::Write for $name {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                let cstr = std::ffi::CString::new(buf).unwrap();
                unsafe {
                    $pyfunc(cstr.as_ptr() as *const std::os::raw::c_char);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
    };
}
make_python_stdio!(PythonStdout, PySys_WriteStdout);
make_python_stdio!(PythonStderr, PySys_WriteStderr);

pub(crate) fn stdout() -> PythonStdout {
    PythonStdout {}
}

#[allow(dead_code)]
pub(crate) fn stderr() -> PythonStderr {
    PythonStderr {}
}
