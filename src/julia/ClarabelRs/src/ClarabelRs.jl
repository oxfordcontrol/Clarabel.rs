module ClarabelRs 

    using SparseArrays, JSON, Libdl
    #we use the main Clarabel Julia implementation 
    #for things like settings instead of redefining 
    #them in this Rust wrapper module 
    using Clarabel

    function __init__()

        # this incredibly fragile path is where one should expect the 
        # rust static lib to appear if compiled with : 
        # > cargo build --release --features julia
        global libpath = joinpath(@__DIR__, 
                        "../../../../target/release/",
                        "libclarabel." * Libdl.dlext
                )
        global librust = Libdl.dlopen(libpath)
    end

    function reload()
        global libpath
        global librust
        
        # force garbage collection to try to prevent (or at least discourage...)
        # segfaults when releasing old objects with a different build
        GC.gc()
        result = false
        while !result
            result = Libdl.dlclose(librust)
        end 
        librust = Libdl.dlopen(libpath)
    
        println("Clarabel rust lib reloaded at", librust)
    end

    using Pkg

    function _get_clarabelrs_version()
        toml_path = joinpath(@__DIR__,"../Project.toml")
        pkg = Pkg.Types.read_package(toml_path)
        string(pkg.version)
    end

    const SOLVER_NAME    = "ClarabelRs"
    const SOLVER_VERSION = _get_clarabelrs_version()

    solver_name() = SOLVER_NAME
    version()     = SOLVER_VERSION
    
    include("./types.jl")
    include("./interface.jl")

    #MathOptInterface for JuMP/Convex.jl
    include("./MOI_wrapper.jl")


end 
