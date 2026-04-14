use super::{auto::*, qdldl::*};

#[cfg(feature = "faer-sparse")]
use super::faer_ldl::*;
#[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
use super::pardiso::*;

use crate::{
    algebra::{CscMatrix, FloatT, MatrixTriangle},
    solver::{
        core::kktsolvers::direct::{BoxedDirectLDLSolver, DirectLDLSolverReqs},
        CoreSettings,
    },
};

type LDLConstructor<T> =
    fn(&CscMatrix<T>, &[i8], &CoreSettings<T>, Option<Vec<usize>>) -> BoxedDirectLDLSolver<T>;

pub trait LDLConfiguration: FloatT {
    fn get_ldlsolver_config(
        settings: &CoreSettings<Self>,
    ) -> (MatrixTriangle, LDLConstructor<Self>);
}

// The Julia version implements the mapping from user settings to LDL solver
// implementation using dynamic dispatch on ::Val{:qdldl} types of arguments.
// There is no equivalent in Rust, so we get these big match statemen

// The implementation first handles the f64-specific solvers (MKL, Panua Pardiso).
// Since trait specialization doesn't exist in Rust, we use runtime TypeId
// checks within a blanket impl to conditionally enable these solvers only for f64.

impl<T: FloatT> LDLConfiguration for T {
    fn get_ldlsolver_config(settings: &CoreSettings<T>) -> (MatrixTriangle, LDLConstructor<T>) {
        let ldlptr: LDLConstructor<T>;
        let kktshape: MatrixTriangle;
        let case = settings.direct_solve_method.as_str();

        match case {
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            "mkl" | "panua" => {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
                    let ldlptr64: LDLConstructor<f64>;
                    match case {
                        #[cfg(feature = "pardiso-mkl")]
                        "mkl" => {
                            kktshape = MKLPardisoDirectLDLSolver::required_matrix_shape();
                            ldlptr64 =
                                |M, D, S, P| Box::new(MKLPardisoDirectLDLSolver::new(M, D, S, P));
                        }

                        #[cfg(feature = "pardiso-panua")]
                        "panua" => {
                            kktshape = PanuaPardisoDirectLDLSolver::required_matrix_shape();
                            ldlptr64 =
                                |M, D, S, P| Box::new(PanuaPardisoDirectLDLSolver::new(M, D, S, P));
                        }
                        _ => {
                            // since one, but not both, might be enabled
                            panic!("Unrecognized LDL solver type: \"{}\"", case);
                        }
                    }
                    // force cast back to generic FloatT, which should be safe
                    // because FloatT == f64 always here
                    ldlptr = unsafe {
                        let ptr: *const LDLConstructor<f64> = &ldlptr64;
                        std::ptr::read(ptr as *const LDLConstructor<T>)
                    };
                } else {
                    panic!(
                        "LDL solver option \"{}\" not supported for {:?} types",
                        case,
                        std::any::type_name::<T>()
                    );
                }
            }
            "auto" => {
                kktshape = AutoDirectLDLSolver::<T>::required_matrix_shape();
                ldlptr = |M, D, S, P| AutoDirectLDLSolver::new(M, D, S, P);
            }
            "qdldl" => {
                kktshape = QDLDLDirectLDLSolver::<T>::required_matrix_shape();
                ldlptr = |M, D, S, P| Box::new(QDLDLDirectLDLSolver::new(M, D, S, P));
            }
            #[cfg(feature = "faer-sparse")]
            "faer" => {
                kktshape = FaerDirectLDLSolver::<T>::required_matrix_shape();
                ldlptr = |M, D, S, P| Box::new(FaerDirectLDLSolver::new(M, D, S, P));
            }
            _ => {
                panic!("Unrecognized LDL solver type: \"{}\"", case);
            }
        };
        (kktshape, ldlptr)
    }
}
