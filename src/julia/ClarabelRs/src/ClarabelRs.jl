module ClarabelRs 

    using SparseArrays, JSON 
    #we use the main Clarabel Julia implementation 
    #for things like settings instead of redefining 
    #them in this Rust wrapper module 
    using Clarabel

    # this incredibly fragile path is where one should expect the 
    # rust static lib to appear if compiled with : 
    # > cargo build --release --features julia
    const LIBPATH = joinpath(@__DIR__, "../../../../target/release/")
    const LIBRUST = joinpath(LIBPATH,"libclarabel." * Base.Libc.dlext)
    
    include("./types.jl")
    include("./interface.jl")

    #MathOptInterface for JuMP/Convex.jl
    include("./MOI_wrapper.jl")


end 
