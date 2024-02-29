// Provides a Writer to allow for redirection of stdout and stderr streams
// to the ones configured for Python.

use pyo3::ffi::{PyObject_CallMethod, PySys_GetObject, PySys_WriteStderr, PySys_WriteStdout};
use std::ffi::CString;
use std::io::{LineWriter, Write};
use std::os::raw::c_char;

macro_rules! make_python_stdio {
    ($rawtypename:ident, $typename:ident, $pyfunc:ident, $pymodname:literal) => {
        pub(crate) struct $rawtypename {
            pub cbuffer: Vec<u8>,
        }
        impl $rawtypename {
            pub(crate) fn new() -> Self {
                Self {
                    cbuffer: Vec::new(),
                }
            }
        }
        impl Write for $rawtypename {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                //clear internal buffer and then overwrite with the
                //new buffer and a null terminator
                self.cbuffer.clear();
                self.cbuffer.extend_from_slice(buf);
                self.cbuffer.push(0);
                unsafe {
                    $pyfunc(self.cbuffer.as_ptr() as *const c_char);
                }
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                // call the python flush() on sys.$pymodname
                unsafe {
                    let stdout_str = CString::new($pymodname).unwrap();
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

        pub(crate) struct $typename {
            inner: LineWriter<$rawtypename>,
        }

        impl $typename {
            pub(crate) fn new() -> Self {
                Self {
                    inner: LineWriter::new($rawtypename::new()),
                }
            }
        }

        impl Write for $typename {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.inner.write(buf)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                self.inner.flush()
            }
        }
    };
}
make_python_stdio!(
    PythonStdoutRaw,
    PythonStdout,
    PySys_WriteStdout,
    "__stdout__"
);
make_python_stdio!(
    PythonStderrRaw,
    PythonStderr,
    PySys_WriteStderr,
    "__stderr__"
);

pub(crate) fn stdout() -> PythonStdout {
    PythonStdout::new()
}

#[allow(dead_code)]
pub(crate) fn stderr() -> PythonStderr {
    PythonStderr::new()
}
