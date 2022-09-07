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
                        "libclarabel." * Base.Libc.dlext
                )
        global librust = Libdl.dlopen(libpath)
    end

    function reload()
        global libpath
        global librust
        
        result = true
        while result
            result = Libdl.dlclose(librust)
        end 
        librust = Libdl.dlopen(libpath)
    
        println("Clarabel rust lib reloaded at", librust)
    end
    
    
    include("./types.jl")
    include("./interface.jl")

    #MathOptInterface for JuMP/Convex.jl
    include("./MOI_wrapper.jl")


end 
