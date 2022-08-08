module ClarabelRs 

    using SparseArrays
    #we use the main Clarabel Julia implementation 
    #for things like settings instead of redefining 
    #them in this Rust wrapper module 
    using Clarabel

    const LIBPATH = joinpath(@__DIR__, "../../../target/release/")
    const LIBRUST = "libclarabel." * Base.Libc.dlext
    
    function path_config()
        #configure paths the compiled rust library
        if LIBPATH ∉ Base.DL_LOAD_PATH
            push!(Base.DL_LOAD_PATH, LIBPATH)
        end
    end

    path_config();

    include("./types.jl")
    include("./interface.jl")

end 
