use crate::solver::implementations::default::DefaultSettings;

//Default implemenentations only uses the
//core settings, so just typedef to the default
pub type CoreSettings<T> = DefaultSettings<T>;
