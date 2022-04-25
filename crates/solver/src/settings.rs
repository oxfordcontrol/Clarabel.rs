use clarabel_algebra::*;

pub struct Settings<T: FloatT = f64> {

    pub max_iter: u32,
    pub time_limit: T,
    pub verbose: bool,
    pub tol_gap_abs: T,
    pub tol_gap_rel: T,
    pub tol_feas: T,
	pub tol_infeas_abs: T,
	pub tol_infeas_rel: T,
    pub max_step_fraction: T,

	// data equilibration
	pub equilibrate_enable: bool,
	pub equilibrate_max_iter: u32,
	pub equilibrate_min_scaling: T,
	pub equilibrate_max_scaling: T,

    // can be :qdldl or :mkl
    pub direct_kkt_solver: bool,
    //pub direct_solve_method: Symbol,   PJG:Add this later

    // static regularization parameters
    pub static_regularization_enable: bool,
    pub static_regularization_eps: T,

    // dynamic regularization parameters
    pub dynamic_regularization_enable: bool,
    pub dynamic_regularization_eps: T,
    pub dynamic_regularization_delta: T,

    // iterative refinement (for QDLDL)
    pub iterative_refinement_enable: bool,
    pub iterative_refinement_reltol: T,
    pub iterative_refinement_abstol: T,
    pub iterative_refinement_max_iter: u32,
    pub iterative_refinement_stop_ratio: T,
}


impl<T: FloatT + std::convert::From<f32>> Settings<T> {
    pub fn new() -> Self
    {
    Self {
        max_iter: 50,
        time_limit: T::zero(),
        verbose:  true,
        tol_gap_abs:  1e-8f32.into(),
        tol_gap_rel:  1e-8f32.into(),
        tol_feas:  1e-5f32.into(),
        tol_infeas_abs:  1e-8f32.into(),
        tol_infeas_rel:  1e-8f32.into(),
        max_step_fraction:  0.99f32.into(),

        // data equilibration
        equilibrate_enable:  true,
        equilibrate_max_iter: 10,
        equilibrate_min_scaling:  1e-4f32.into(),
        equilibrate_max_scaling:  1e+4f32.into(),

        // can be :qdldl or :mkl
        direct_kkt_solver:  true,   //indirect not yet supported
        //direct_solve_method:  :qdldl,

        // static regularization parameters
        static_regularization_enable:  true,
        static_regularization_eps:  1e-8f32.into(),

        // dynamic regularization parameters
        dynamic_regularization_enable:  true,
        dynamic_regularization_eps:  1e-13f32.into(),
        dynamic_regularization_delta:  2e-7f32.into(),

        // iterative refinement (for QDLDL)
        iterative_refinement_enable:  true,
        iterative_refinement_reltol:  1e-10f32.into(),
        iterative_refinement_abstol: 1e-10f32.into(),
        iterative_refinement_max_iter: 10,
        iterative_refinement_stop_ratio: 2f32.into(),
    }
}
}
