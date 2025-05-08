use crate::{algebra::*, solver::*};

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
    #[serde(default)]
    pub settings: DefaultSettings<T>,
}

impl<T> SolverJSONReadWrite<T> for DefaultSolver<T>
where
    T: FloatT + DeserializeOwned + Serialize,
{
    fn save_to_file(&self, file: &mut File) -> Result<(), io::Error> {
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

    fn load_from_file(
        file: &mut File,
        settings: Option<DefaultSettings<T>>,
    ) -> Result<Self, SolverError> {
        // read file
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;

        // Parse JSON and convert any serde_json::Error to SolverError::JsonError
        let json_data: Result<JsonProblemData<T>, _> = serde_json::from_str(&buffer);
        let mut json_data = match json_data {
            Ok(data) => data,
            Err(e) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("JSON parsing error: {}", e),
                )
                .into())
            }
        };

        // restore sanitized settings to their (likely) original values
        desanitize_settings(&mut json_data.settings);

        // create a solver object
        let P = json_data.P;
        let q = json_data.q;
        let A = json_data.A;
        let b = json_data.b;
        let cones = json_data.cones;
        let settings = settings.unwrap_or(json_data.settings);

        // Convert SolverError to io::Error
        Self::new(&P, &q, &A, &b, &cones, settings)
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
