macro_rules! printinfo {
    ($($tokens: tt)*) => {
        println!("cargo:warning=\r\x1b[36;1m   {}", format!($($tokens)*))
    }
}

fn main() {
    config_python_blas();
}

fn config_python_blas() {
    // cfg(sdp_pyblas) is used to indicate that BLAS/LAPACK functions
    // should be taken from Python's scipy.  It will only be defined
    // for python builds that do not link to one of the blas/lapack
    // libraries provided by blas-src and lapack-src.
    println!("cargo:rustc-check-cfg=cfg(sdp_pyblas)");

    if cfg!(not(feature = "python")) {
        return;
    }

    if !cfg!(any(
        feature = "sdp-accelerate",
        feature = "sdp-netlib",
        feature = "sdp-openblas",
        feature = "sdp-mkl",
        feature = "sdp-r"
    )) {
        println!("cargo:rustc-cfg=sdp_pyblas");
        printinfo!("Python: compiling with python blas from scipy");
    } else {
        printinfo!("Python: compiling with local blas/lapack libraries");
    }
}
