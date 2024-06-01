use crate::{
    algebra::*,
    solver::{core::SolverJSONReadWrite, DefaultSettings, DefaultSolver, SupportedConeT},
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::io::Write;
use std::{fs::File, io, io::Read};

// A struct very similar to the problem data, but containing only
// the data types provided by the user (i.e. no internal types).

#[derive(Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
struct JsonProblemData<T: FloatT> {
    pub P: CscMatrix<T>,
    pub q: Vec<T>,
    pub A: CscMatrix<T>,
    pub b: Vec<T>,
    pub cones: Vec<SupportedConeT<T>>,
    pub settings: DefaultSettings<T>,
}

impl<T> SolverJSONReadWrite for DefaultSolver<T>
where
    T: FloatT + DeserializeOwned + Serialize,
{
    fn write_to_file(&self, file: &mut File) -> Result<(), io::Error> {
        let mut json_data = JsonProblemData {
            P: self.data.P.clone(),
            q: self.data.q.clone(),
            A: self.data.A.clone(),
            b: self.data.b.clone(),
            cones: self.data.cones.clone(),
            settings: self.settings.clone(),
        };

        // restore scaling to original
        let dinv = &self.data.equilibration.dinv;
        let einv = &self.data.equilibration.einv;
        let c = &self.data.equilibration.c;

        json_data.P.lrscale(dinv, dinv);
        json_data.q.hadamard(dinv);
        json_data.P.scale(c.recip());
        json_data.q.scale(c.recip());

        json_data.A.lrscale(einv, dinv);
        json_data.b.hadamard(einv);

        // sanitize settings to remove values that
        // can't be serialized, i.e. infs
        sanitize_settings(&mut json_data.settings);

        // write to file
        let json = serde_json::to_string(&json_data)?;
        file.write_all(json.as_bytes())?;

        Ok(())
    }

    fn read_from_file(file: &mut File) -> Result<Self, io::Error> {
        // read file
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let mut json_data: JsonProblemData<T> = serde_json::from_str(&buffer)?;

        // restore sanitized settings to their (likely) original values
        desanitize_settings(&mut json_data.settings);

        // create a solver object
        let P = json_data.P;
        let q = json_data.q;
        let A = json_data.A;
        let b = json_data.b;
        let cones = json_data.cones;
        let settings = json_data.settings;
        let solver = Self::new(&P, &q, &A, &b, &cones, settings);

        Ok(solver)
    }
}

fn sanitize_settings<T: FloatT>(settings: &mut DefaultSettings<T>) {
    if settings.time_limit == f64::INFINITY {
        settings.time_limit = f64::MAX;
    }
}

fn desanitize_settings<T: FloatT>(settings: &mut DefaultSettings<T>) {
    if settings.time_limit == f64::MAX {
        settings.time_limit = f64::INFINITY;
    }
}

#[test]
fn test_json_io() {
    use crate::solver::IPSolver;
    use std::io::{Seek, SeekFrom};

    let P = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![2.0],
    };
    let q = [1.0];
    let A = CscMatrix {
        m: 1,
        n: 1,
        colptr: vec![0, 1],
        rowval: vec![0],
        nzval: vec![-1.0],
    };
    let b = [-2.0];
    let cones = vec![crate::solver::SupportedConeT::NonnegativeConeT(1)];

    let settings = crate::solver::DefaultSettingsBuilder::default()
        .build()
        .unwrap();

    let mut solver = crate::solver::DefaultSolver::<f64>::new(&P, &q, &A, &b, &cones, settings);
    solver.solve();

    // write the problem to a file
    let mut file = tempfile::tempfile().unwrap();
    solver.write_to_file(&mut file).unwrap();

    // read the problem from the file
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut solver2 = crate::solver::DefaultSolver::<f64>::read_from_file(&mut file).unwrap();
    solver2.solve();
    assert_eq!(solver.solution.x, solver2.solution.x);
}
