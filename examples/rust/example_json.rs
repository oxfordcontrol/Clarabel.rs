#![allow(non_snake_case)]
use clarabel::solver::*;
use std::env;
use std::fs::File;
use std::path::PathBuf;

fn main() {
    let filename = "Foundation_3D_1099.json";

    // Get the path to the crate root using the CARGO_MANIFEST_DIR environment variable
    let cargo_dir = env::var("CARGO_MANIFEST_DIR").expect("Failed to get CARGO_MANIFEST_DIR");
    let data_path = PathBuf::from(cargo_dir).join("examples").join("data");

    // now I have the path to the file
    let filename = data_path.join(filename);
    let mut file = File::open(&filename).unwrap();

    // override the settings in the loaded file
    let settings = DefaultSettings {
        input_sparse_dropzeros: true,
        iterative_refinement_enable: false,
        direct_solve_method: "faer".to_string(),
        ..DefaultSettings::default()
    };

    let mut solver = DefaultSolver::<f64>::load_from_file(&mut file, Some(settings)).unwrap();
    solver.solve();

    // to write the back to a new file

    // let outfile = "./examples/data/output.json";
    // let mut file = File::create(outfile).unwrap();
    // solver.save_to_file(&mut file).unwrap();
}
