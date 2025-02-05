//! Types for managing solver output to various targets.
//!

use std::fs::File;
use std::io::{Error, ErrorKind, Result, Write};

#[cfg(not(feature = "python"))]
#[allow(unused_imports)]
pub(crate) use std::io::{stderr, stdout, Stdout};

// configure python specific stdout and stdin streams
// when compiled with the python feature.   This avoids
// problems when running within python notebooks etc.
#[cfg(feature = "python")]
#[allow(unused_imports)]
pub(crate) use crate::python::io::{stderr, stdout, Stdout};

/// Container for managing multiple print targets
pub(crate) enum PrintTarget {
    Stdout(Stdout),
    File(File),
    Buffer(Vec<u8>),
    Stream(Box<dyn Write + Send + Sync>), // Supports any stream that implements `Write`
}

impl<'a> From<&'a mut PrintTarget> for Box<&'a mut dyn Write> {
    fn from(target: &'a mut PrintTarget) -> Self {
        match target {
            PrintTarget::Stdout(ref mut stdout) => Box::new(stdout),
            PrintTarget::File(ref mut file) => Box::new(file),
            PrintTarget::Stream(ref mut stream) => Box::new(stream),
            PrintTarget::Buffer(ref mut buffer) => Box::new(buffer),
        }
    }
}

impl std::fmt::Debug for PrintTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrintTarget::Stdout(_) => write!(f, "PrintTarget::Stdout"),
            PrintTarget::File(_) => write!(f, "PrintTarget::File"),
            PrintTarget::Buffer(_) => write!(f, "PrintTarget::Buffer"),
            PrintTarget::Stream(_) => write!(f, "PrintTarget::Stream"),
        }
    }
}

impl Default for PrintTarget {
    fn default() -> Self {
        PrintTarget::Stdout(self::stdout())
    }
}

impl Write for PrintTarget {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        match self {
            PrintTarget::Stdout(stdout) => stdout.write(buf),
            PrintTarget::File(file) => file.write(buf),
            PrintTarget::Buffer(buffer) => {
                buffer.extend_from_slice(buf);
                Ok(buf.len())
            }
            PrintTarget::Stream(stream) => stream.write(buf),
        }
    }

    fn flush(&mut self) -> Result<()> {
        match self {
            PrintTarget::Stdout(stdout) => stdout.flush(),
            PrintTarget::File(file) => file.flush(),
            PrintTarget::Buffer(_) => Ok(()),
            PrintTarget::Stream(stream) => stream.flush(),
        }
    }
}

/// Trait implemented by solvers that allow configurable print targets
pub trait ConfigurablePrintTarget {
    /// redirect print output to stdout
    fn print_to_stdout(&mut self);
    /// redirect print output to a file
    fn print_to_file(&mut self, file: File);
    /// redirect print output to a stream
    fn print_to_stream(&mut self, stream: Box<dyn Write + Send + Sync>);
    /// redirect print output to an internal buffer
    fn print_to_buffer(&mut self);
    /// get the contents of the internal print buffer
    fn get_print_buffer(&mut self) -> Result<String>;
    /// get the current print target
    fn print_target(&mut self) -> &dyn Write;
}

impl ConfigurablePrintTarget for PrintTarget {
    fn print_to_stdout(&mut self) {
        *self = PrintTarget::Stdout(self::stdout());
    }

    fn print_to_file(&mut self, file: File) {
        *self = PrintTarget::File(file);
    }

    fn print_to_stream(&mut self, stream: Box<dyn Write + Send + Sync>) {
        *self = PrintTarget::Stream(stream);
    }

    fn print_to_buffer(&mut self) {
        *self = PrintTarget::Buffer(Vec::new());
    }

    fn get_print_buffer(&mut self) -> std::io::Result<String> {
        match self {
            PrintTarget::Buffer(buffer) => Ok(String::from_utf8_lossy(buffer).to_string()),
            _ => Err(Error::new(
                ErrorKind::Other,
                "Print buffering is not configured.",
            )),
        }
    }
    fn print_target(&mut self) -> &dyn Write {
        self
    }
}
