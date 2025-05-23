#![allow(non_snake_case)]

use clarabel::{algebra::*, io::ConfigurablePrintTarget, solver::*};

#[allow(clippy::type_complexity)]
fn test_print_solver() -> DefaultSolver<f64> {
    let P = CscMatrix::identity(1);
    let c = [0.];
    let A = CscMatrix::identity(1);
    let b = [1.];
    let cones = [NonnegativeConeT(1)];
    let settings = DefaultSettings::default();
    DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap()
}

#[test]
fn test_print_to_stdout() {
    let mut solver = test_print_solver();
    solver.print_to_stdout();
    solver.solve();
}

#[test]
fn test_print_to_buffer() {
    let mut solver = test_print_solver();
    solver.print_to_buffer();
    solver.solve();
    let result = solver.get_print_buffer().unwrap();
    assert!(result.contains("Clarabel.rs"));
}

#[test]
fn test_print_to_file() {
    use std::io::{Read, Seek};

    let mut solver = test_print_solver();
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut file2 = file.reopen().unwrap();
    solver.print_to_file(file.into_file());
    solver.solve();

    file2.seek(std::io::SeekFrom::Start(0)).unwrap();
    let mut result = String::new();
    file2.read_to_string(&mut result).unwrap();
    assert!(result.contains("Clarabel.rs"));
}

#[test]
fn test_print_to_stream() {
    use std::io::{Read, Seek};

    let mut solver = test_print_solver();
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut file2 = file.reopen().unwrap();
    let stream = Box::new(file.into_file());

    solver.print_to_stream(stream);
    solver.solve();

    file2.seek(std::io::SeekFrom::Start(0)).unwrap();
    let mut result = String::new();
    file2.read_to_string(&mut result).unwrap();
    assert!(result.contains("Clarabel.rs"));
}

#[test]
fn test_print_to_sink() {
    let mut solver = test_print_solver();
    solver.print_to_sink();
    solver.solve();
    // no output
}
