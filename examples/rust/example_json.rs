#![allow(non_snake_case)]
use clarabel::solver::*;
use std::fs::File;

fn main() {
    // HS35 is a small problem QP problem
    // from the Maros-Meszaros test set

    let filename = "./examples/data/hs35.json";
    let mut file = File::open(filename).unwrap();
    let mut solver = DefaultSolver::<f64>::load_from_file(&mut file, None).unwrap();
    solver.solve();

    // to write the back to a new file

    // let outfile = "./examples/data/output.json";
    // let mut file = File::create(outfile).unwrap();
    // solver.save_to_file(&mut file).unwrap();
}
