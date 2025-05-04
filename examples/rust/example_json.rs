#![allow(non_snake_case)]
use clarabel::solver::*;
use std::env;
use std::fs::File;
use std::path::PathBuf;

fn main() {
    // HS35 is a small problem QP problem
    // from the Maros-Meszaros test set
    let filename = "hs35.json";

    // Get the path to the crate root using the CARGO_MANIFEST_DIR environment variable
    let cargo_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let data_path = PathBuf::from(cargo_dir).join("examples").join("data");

    // now I have the path to the file
    let filename = data_path.join(filename);
    let mut file = File::open(&filename).unwrap();

    let mut solver = DefaultSolver::<f64>::load_from_file(&mut file, None).unwrap();
    solver.solve();

    // to write the back to a new file

    // let outfile = "./examples/data/output.json";
    // let mut file = File::create(outfile).unwrap();
    // solver.save_to_file(&mut file).unwrap();
}
