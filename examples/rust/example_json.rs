#![allow(non_snake_case)]
use clarabel::solver::*;
use std::env;
use std::path::PathBuf;

fn main() {
    let filename = "hs35.json";

    // Get the path to the crate root using the CARGO_MANIFEST_DIR environment variable
    let cargo_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let data_path = PathBuf::from(cargo_dir).join("examples").join("data");

    // now I have the path to the file
    let filename = data_path.join(filename);

    let settings = DefaultSettingsBuilder::default().build().unwrap();

    let mut solver = DefaultSolver::<f64>::load_from_file(&filename, Some(settings)).unwrap();
    solver.solve();

    // to write back to a new file

    // let outfile = "./examples/data/output.json";
    // solver.save_to_file(&outfile).unwrap();
}
