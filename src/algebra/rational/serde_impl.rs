//! `serde::Serialize` / `Deserialize` for [`RationalReal`].
//!
//! `RationalReal` is an arena handle, so we cannot serialize the
//! `u32` index directly — the receiving end's arena does not contain
//! that slot. Instead we serialize the underlying `BigRational` value
//! when finite, or a special string marker for the IEEE sentinels
//! ("+inf" / "-inf" / "NaN") so that propagation-aware semantics
//! survive the round trip. On deserialize we either push the value
//! into the destination thread's arena (returning a finite-tagged
//! handle) or rebuild the corresponding sentinel handle.
//!
//! Format: a tagged enum `RationalReal::Finite(BigRational)` /
//! `RationalReal::PosInf` / `RationalReal::NegInf` / `RationalReal::Nan`.
//! In JSON this looks like `{"Finite":[numer_str,denom_str]}` or
//! `"PosInf"` / `"NegInf"` / `"Nan"`. This is a **breaking change**
//! relative to the previous format (raw `BigRational`) but the
//! previous format had no way to represent sentinels.

use super::arena;
use super::real::RationalReal;
use num_rational::BigRational;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// On-the-wire form. Mirrors the four-class tag of the handle.
#[derive(Serialize, Deserialize)]
enum RationalRealRepr {
    Finite(BigRational),
    PosInf,
    NegInf,
    Nan,
}

impl Serialize for RationalReal {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        if arena::is_pos_inf_handle(self.0) {
            return RationalRealRepr::PosInf.serialize(ser);
        }
        if arena::is_neg_inf_handle(self.0) {
            return RationalRealRepr::NegInf.serialize(ser);
        }
        if arena::is_nan_handle(self.0) {
            return RationalRealRepr::Nan.serialize(ser);
        }
        // Finite: clone out of the arena and wrap in the Finite variant.
        // The Serialize path is not in the solver hot loop; the cost is
        // one BigInt clone pair, paid once per output value.
        arena::with(self.0, |r| {
            RationalRealRepr::Finite(r.clone()).serialize(ser)
        })
    }
}

impl<'de> Deserialize<'de> for RationalReal {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let repr = RationalRealRepr::deserialize(de)?;
        Ok(match repr {
            RationalRealRepr::Finite(r) => RationalReal::from_bigrational(r),
            RationalRealRepr::PosInf => RationalReal(arena::POS_INF_HANDLE),
            RationalRealRepr::NegInf => RationalReal(arena::NEG_INF_HANDLE),
            RationalRealRepr::Nan => RationalReal(arena::NAN_HANDLE),
        })
    }
}
