# -------------------------------------
# API constructor 
#--------------------------------------
function Solver(
    P::SparseMatrixCSC{Float64,Int64},
    c::Vector{Float64},
    A::SparseMatrixCSC{Float64,Int64},
    b::Vector{Float64},
    cone_types::Vector{<:Clarabel.SupportedCone},
    settings::Clarabel.Settings{Float64}
) 
    s = solver_new_jlrs(P,c,A,b,cone_types,settings)
    return s
end

function solve!(solver::Solver)

    #this is the solution as a SolutionJLRS
    sol_jlrs = solver_solve_jlrs(solver)

    DefaultSolution(sol_jlrs)

end

function get_info(solver::Solver)

    info = solver_get_info_jlrs(solver::Solver)
    return info

end


# -------------------------------------
# Wrappers for rust-side interface  
#--------------------------------------

function solver_new_jlrs(P,q,A,b,cones,settings)


    # first flatten the cones to three primitive arrays 
    (cone_enums, cone_ints, cone_floats) = ccall_cones_to_arrays(cones)

    ptr = ccall(Libdl.dlsym(librust,:solver_new_jlrs),Ptr{Cvoid},
        (
            Ref{CscMatrixJLRS},         #P
            Ref{VectorJLRS{Float64}},   #q
            Ref{CscMatrixJLRS},         #A
            Ref{VectorJLRS{Float64}},   #b
            Ref{VectorJLRS{UInt8}},     #cone_enums
            Ref{VectorJLRS{UInt64}},    #cone_ints
            Ref{VectorJLRS{Float64}},   #cone_floats
            Cstring                     #json_settings
        ),
            CscMatrixJLRS(P),           #P
            VectorJLRS(q),              #q
            CscMatrixJLRS(A),           #A
            VectorJLRS(b),              #b
            VectorJLRS(cone_enums),     #cone_enums
            VectorJLRS(cone_ints),      #cone_ints
            VectorJLRS(cone_floats),    #cone_floats
            serialize(settings),        #serialized settings
        )

    return Solver{Float64}(ptr)
end 


function solver_solve_jlrs(solver::Solver)

    ccall(Libdl.dlsym(librust,:solver_solve_jlrs),SolutionJLRS,
        (Ptr{Cvoid},), solver.ptr)

end 

function solver_get_info_jlrs(solver::Solver)

    ccall(Libdl.dlsym(librust,:solver_get_info_jlrs),Clarabel.DefaultInfo{Float64},
    (Ptr{Cvoid},), solver.ptr)
    
end


function solver_drop_jlrs(solver::Solver)
    ccall(Libdl.dlsym(librust,:solver_drop_jlrs),Cvoid,
        (Ptr{Cvoid},), solver.ptr)

end 


# -------------------------------------
# functions for passing cones and settings 
# -------------------------------------

# it is not obvious at all how to pass data through 
# ccall for a data-carrying enum type in rust.   This 
# makes it very difficult to pass the `cones` object 
# directly.   Here we make an enum for the different 
# cone types, with a complementary enum type on 
# the rust side with equivalent base types and values 
# We will pass three arrays to rust : the enum value,
# an integer (for dimensions) and a float (for powers)
# Every cone needs at least of one these values.  
# Values not needed for a particular cone get a zero 
# placeholder 

function ccall_cones_to_arrays(cones::Vector{Clarabel.SupportedCone})

    cone_enums  = zeros(UInt8,length(cones))
    cone_ints   = zeros(UInt64,length(cones))
    cone_floats = zeros(Float64,length(cones))

    for (i,cone) in enumerate(cones)

        if isa(cone, Clarabel.ZeroConeT)
            cone_enums[i] = UInt8(ZeroConeT::ConeEnumJLRS)
            cone_ints[i]  = cone.dim;

        elseif isa(cone, Clarabel.NonnegativeConeT)
            cone_enums[i] = UInt8(NonnegativeConeT::ConeEnumJLRS) 
            cone_ints[i]  = cone.dim;

        elseif isa(cone, Clarabel.SecondOrderConeT)
            cone_enums[i] = UInt8(SecondOrderConeT::ConeEnumJLRS) 
            cone_ints[i]  = cone.dim;

        elseif isa(cone, Clarabel.ExponentialConeT)
            cone_enums[i] = UInt8(ExponentialConeT::ConeEnumJLRS) 

        elseif isa(cone, Clarabel.PowerConeT)
            cone_enums[i] = UInt8(PowerConeT::ConeEnumJLRS) 
            cone_floats[i] = cone.α

        elseif isa(cone, Clarabel.PSDTriangleConeT)
            cone_enums[i] = UInt8(PSDTriangleConeT::ConeEnumJLRS) 
            cone_ints[i]  = cone.dim;
        else 
            error("Cone type ", typeof(cone), " is not supported through this interface.");
        end
    end 

    return (cone_enums, cone_ints, cone_floats)
end 


# It is not straightforward to pass the Julia settings structure 
# to Rust, primarily because the String and Symbol types are 
# not mutually intelligible.   This allows passing via JSON 
# serialization (in Julia) / deserialization (in Rust)

function serialize(settings::Clarabel.Settings)

    # JSON sets inf values to "null".  Make a copy so that 
    #we can make inf value safe to tranmit.   
    settings = deepcopy(settings)

    if(!isfinite(settings.time_limit))
        settings.time_limit = floatmax(Float64)
    end

    JSON.json(settings)

end
