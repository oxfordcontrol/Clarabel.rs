// Provides a Writer to allow for redirection of stdout and stderr streams
// to the ones configured for Python.

use pyo3::ffi::{c_str, PySys_WriteStderr, PySys_WriteStdout};
use pyo3::prelude::*;
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
                Python::with_gil(|_py| unsafe {
                    $pyfunc(self.cbuffer.as_ptr() as *const c_char);
                });
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                // call the python flush() on sys.$pymodname
                Python::with_gil(|py| -> std::io::Result<()> {
                    py.run(
                        c_str!(std::concat!("import sys; sys.", $pymodname, ".flush()")),
                        None,
                        None,
                    )
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                    Ok(())
                })
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
make_python_stdio!(PythonStdoutRaw, PythonStdout, PySys_WriteStdout, "stdout");
make_python_stdio!(PythonStderrRaw, PythonStderr, PySys_WriteStderr, "stderr");

pub(crate) fn stdout() -> PythonStdout {
    PythonStdout::new()
}

#[allow(dead_code)]
pub(crate) fn stderr() -> PythonStderr {
    PythonStderr::new()
}

#[allow(unused_imports)]
pub(crate) use PythonStderr as Stderr;
pub(crate) use PythonStdout as Stdout;
