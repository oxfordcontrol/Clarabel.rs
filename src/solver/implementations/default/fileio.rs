use crate::{algebra::*, solver::*};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, Read, Write},
    path::Path,
};

// A struct very similar to the problem data, but containing only
// the data types provided by the user (i.e. no internal types).

#[derive(Serialize, Deserialize)]
#[serde(bound = "T: Serialize + DeserializeOwned")]
struct SerializedProblemData<T: FloatT> {
    pub P: CscMatrix<T>,
    pub q: Vec<T>,
    pub A: CscMatrix<T>,
    pub b: Vec<T>,
    pub cones: Vec<SupportedConeT<T>>,
    #[serde(default)]
    pub settings: DefaultSettings<T>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SerializationFormat {
    Json,
    Bincode,
}

impl<T> SolverSerializedReadWrite<T> for DefaultSolver<T>
where
    T: FloatT + DeserializeOwned + Serialize,
{
    fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let mut serialized_data = SerializedProblemData {
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

        serialized_data.P.lrscale(dinv, dinv);
        serialized_data.q.hadamard(dinv);
        serialized_data.P.scale(c.recip());
        serialized_data.q.scale(c.recip());

        serialized_data.A.lrscale(einv, dinv);
        serialized_data.b.hadamard(einv);

        // sanitize settings to remove values that
        // can't be serialized, i.e. infs
        sanitize_settings(&mut serialized_data.settings);

        let path = path.as_ref();
        let format = determine_format_from_path(path)?;
        let mut file = File::create(path)?;

        match format {
            SerializationFormat::Json => serialize_to_json(&serialized_data, &mut file)?,
            SerializationFormat::Bincode => serialize_to_bincode(&serialized_data, &mut file)?,
        }

        file.flush()?;

        Ok(())
    }

    fn load_from_file<P: AsRef<Path>>(
        path: P,
        settings: Option<DefaultSettings<T>>,
    ) -> Result<Self, SolverError> {
        //
        let path = path.as_ref();
        let mut file = File::open(path)?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let format = determine_format_from_path(path).unwrap_or(SerializationFormat::Json);

        let mut serialized_data = match format {
            SerializationFormat::Json => deserialize_from_json::<T>(&buffer)?,
            SerializationFormat::Bincode => deserialize_from_bincode::<T>(&buffer)?,
        };

        // restore sanitized settings to their (likely) original values
        desanitize_settings(&mut serialized_data.settings);

        // create a solver object
        let P = serialized_data.P;
        let q = serialized_data.q;
        let A = serialized_data.A;
        let b = serialized_data.b;
        let cones = serialized_data.cones;
        let settings = settings.unwrap_or(serialized_data.settings);

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

fn determine_format_from_path(path: &Path) -> Result<SerializationFormat, io::Error> {
    let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");

    match ext {
        "json" => Ok(SerializationFormat::Json),
        "cbin" => Ok(SerializationFormat::Bincode),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput, //InvalidFilename is msrv 1.87
            format!("Unsupported file extension: {ext}"),
        )),
    }
}

fn serialize_to_json<T: FloatT + Serialize + DeserializeOwned>(
    data: &SerializedProblemData<T>,
    writer: &mut impl Write,
) -> Result<(), io::Error> {
    serde_json::to_writer(writer, data).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("JSON serialization error: {err}"),
        )
    })
}

fn serialize_to_bincode<T: FloatT + Serialize + DeserializeOwned>(
    data: &SerializedProblemData<T>,
    writer: &mut impl Write,
) -> Result<(), io::Error> {
    bincode::serialize_into(writer, data).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bincode serialization error: {err}"),
        )
    })
}

fn deserialize_from_bincode<T: FloatT + Serialize + DeserializeOwned>(
    bytes: &[u8],
) -> Result<SerializedProblemData<T>, io::Error> {
    bincode::deserialize(bytes).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bincode serialization error: {err}"),
        )
    })
}

fn deserialize_from_json<T: FloatT + Serialize + DeserializeOwned>(
    bytes: &[u8],
) -> Result<SerializedProblemData<T>, io::Error> {
    serde_json::from_slice::<SerializedProblemData<T>>(bytes).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("JSON deserialization error: {err}"),
        )
    })
}
