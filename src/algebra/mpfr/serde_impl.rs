use super::precision::default_precision;
use super::real::MpfrFloat;
use rug::Float as RugFloat;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

impl Serialize for MpfrFloat {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let prec = self.prec();
        let s = format!("{}", self.as_rug());
        (prec, s).serialize(ser)
    }
}

impl<'de> Deserialize<'de> for MpfrFloat {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let (prec, s): (u32, String) = Deserialize::deserialize(de)?;
        let prec = if prec == 0 { default_precision() } else { prec };
        match RugFloat::parse(&s) {
            Ok(incomplete) => Ok(MpfrFloat(super::arena::push(RugFloat::with_val(prec, incomplete)))),
            Err(e) => Err(serde::de::Error::custom(format!(
                "MpfrFloat deserialize: {e}"
            ))),
        }
    }
}
