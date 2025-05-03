//! Types for managing solver output to various targets.
//!

use std::fs::File;
use std::io::{Error, ErrorKind, Result, Sink, Write};

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
    Sink(Sink),
}

impl<'a> From<&'a mut PrintTarget> for &'a mut dyn Write {
    fn from(target: &'a mut PrintTarget) -> Self {
        match target {
            PrintTarget::Stdout(ref mut stdout) => stdout,
            PrintTarget::File(ref mut file) => file,
            PrintTarget::Stream(ref mut stream) => stream.as_mut(),
            PrintTarget::Buffer(ref mut buffer) => buffer,
            PrintTarget::Sink(ref mut sink) => sink,
        }
    }
}

impl Clone for PrintTarget {
    fn clone(&self) -> Self {
        match self {
            PrintTarget::Stdout(_) => PrintTarget::Stdout(self::stdout()),
            PrintTarget::File(file) => PrintTarget::File(file.try_clone().unwrap()),
            PrintTarget::Buffer(buffer) => PrintTarget::Buffer(buffer.clone()),
            //arbitary stream cloning is not supported
            PrintTarget::Stream(_) => PrintTarget::Sink(std::io::sink()),
            PrintTarget::Sink(_) => PrintTarget::Sink(std::io::sink()),
        }
    }
}

// explicit debug implementation because Python stdout does not implement Debug
impl std::fmt::Debug for PrintTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrintTarget::Stdout(_) => write!(f, "PrintTarget::Stdout"),
            PrintTarget::File(_) => write!(f, "PrintTarget::File"),
            PrintTarget::Buffer(_) => write!(f, "PrintTarget::Buffer"),
            PrintTarget::Stream(_) => write!(f, "PrintTarget::Stream"),
            PrintTarget::Sink(_) => write!(f, "PrintTarget::Sink"),
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
            PrintTarget::Sink(sink) => sink.write(buf),
        }
    }

    fn flush(&mut self) -> Result<()> {
        match self {
            PrintTarget::Stdout(stdout) => stdout.flush(),
            PrintTarget::File(file) => file.flush(),
            PrintTarget::Buffer(_) => Ok(()),
            PrintTarget::Stream(stream) => stream.flush(),
            PrintTarget::Sink(sink) => sink.flush(),
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
    /// redirect print output to sink (no output)
    fn print_to_sink(&mut self);
    /// redirect print output to an internal buffer
    fn print_to_buffer(&mut self);
    /// get the contents of the internal print buffer
    fn get_print_buffer(&mut self) -> Result<String>;
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
    fn print_to_sink(&mut self) {
        *self = PrintTarget::Sink(std::io::sink());
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
}

#[test]
fn test_print_target_from() {
    use std::io::Cursor;

    // stdout
    let mut target = PrintTarget::Stdout(stdout());
    let writer: &mut dyn Write = (&mut target).into();
    assert!(writer.write(b"foo").is_ok());

    // file
    let file = tempfile::tempfile().unwrap();
    let mut target = PrintTarget::File(file);
    let writer: &mut dyn Write = (&mut target).into();
    assert!(writer.write(b"foo").is_ok());

    // buffer
    let mut target = PrintTarget::Buffer(Vec::new());
    let writer: &mut dyn Write = (&mut target).into();
    assert!(writer.write(b"foo").is_ok());

    // stream
    let mut target = PrintTarget::Stream(Box::new(Cursor::new(Vec::new())));
    let writer: &mut dyn Write = (&mut target).into();
    assert!(writer.write(b"foo").is_ok());

    // sink
    let mut target = PrintTarget::Sink(std::io::sink());
    let writer: &mut dyn Write = (&mut target).into();
    assert!(writer.write(b"foo").is_ok());
}

#[test]
fn test_print_target_debug_and_clone() {
    use std::io::Cursor;
    //stdout
    let target = PrintTarget::Stdout(stdout());
    assert_eq!(format!("{:?}", target.clone()), "PrintTarget::Stdout");

    //file
    let file = tempfile::tempfile().unwrap();
    let target = PrintTarget::File(file);
    assert_eq!(format!("{:?}", target.clone()), "PrintTarget::File");

    //buffer
    let target = PrintTarget::Buffer(Vec::new());
    assert_eq!(format!("{:?}", target.clone()), "PrintTarget::Buffer");

    //stream
    let target = PrintTarget::Stream(Box::new(Cursor::new(Vec::new())));
    assert_eq!(format!("{target:?}"), "PrintTarget::Stream");
    assert_eq!(format!("{:?}", target.clone()), "PrintTarget::Sink");

    // sink
    let target = PrintTarget::Sink(std::io::sink());
    let cloned_target = target.clone();
    assert_eq!(format!("{cloned_target:?}"), "PrintTarget::Sink");
}
