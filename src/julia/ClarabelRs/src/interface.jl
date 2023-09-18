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

function print_timers(solver::Solver)

    solver_print_timers_jlrs(solver::Solver)

end

# -------------------------------------
# Wrappers for rust-side interface  
#--------------------------------------

function solver_new_jlrs(P,q,A,b,cones,settings)


    # first flatten the cones to three primitive arrays 
    (cone_data) = ccall_cones_to_array(cones)

    ptr = ccall(Libdl.dlsym(librust,:solver_new_jlrs),Ptr{Cvoid},
        (
            Ref{CscMatrixJLRS},         #P
            Ref{VectorJLRS{Float64}},   #q
            Ref{CscMatrixJLRS},         #A
            Ref{VectorJLRS{Float64}},   #b
            Ref{VectorJLRS{ConeDataJLRS}},   #cone_data in tagged form
            Cstring                     #json_settings
        ),
            CscMatrixJLRS(P),           #P
            VectorJLRS(q),              #q
            CscMatrixJLRS(A),           #A
            VectorJLRS(b),              #b
            VectorJLRS(cone_data),      #cone data in tagged form
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


function solver_print_timers_jlrs(solver::Solver)

    ccall(Libdl.dlsym(librust,:solver_print_timers_jlrs),Cvoid,
    (Ptr{Cvoid},), solver.ptr)
    
end

function solver_drop_jlrs(solver::Solver)
    ccall(Libdl.dlsym(librust,:solver_drop_jlrs),Cvoid,
        (Ptr{Cvoid},), solver.ptr)

end 


# -------------------------------------
# functions for passing cones and settings 
# -------------------------------------


function ccall_cones_to_array(cones::Vector{Clarabel.SupportedCone})

    rscones = ConeDataJLRS[]
    sizehint!(rscones,length(cones))
    
    for cone in cones

        rscone = begin 
            if isa(cone, Clarabel.ZeroConeT)
                ConeDataJLRS(ZeroConeT::ConeEnumJLRS;
                    int = cone.dim,
                )

            elseif isa(cone, Clarabel.NonnegativeConeT)
                ConeDataJLRS(
                    NonnegativeConeT::ConeEnumJLRS;
                    int = cone.dim,
                )

            elseif isa(cone, Clarabel.SecondOrderConeT)
                ConeDataJLRS(
                    SecondOrderConeT::ConeEnumJLRS;
                    int = cone.dim,
                )

            elseif isa(cone, Clarabel.ExponentialConeT)
                ConeDataJLRS(
                    ExponentialConeT::ConeEnumJLRS
                )

            elseif isa(cone, Clarabel.PowerConeT)
                ConeDataJLRS(
                    PowerConeT::ConeEnumJLRS;
                    float = cone.α
                )

            elseif isa(cone, Clarabel.GenPowerConeT)   
                ConeDataJLRS(
                    GenPowerConeT::ConeEnumJLRS;
                    int = cone.dim2,
                    vec = cone.α
                )

            elseif isa(cone, Clarabel.PSDTriangleConeT)
                ConeDataJLRS(
                    PSDTriangleConeT::ConeEnumJLRS;
                    int = cone.dim,
                )
            else 
                error("Cone type ", typeof(cone), " is not supported through this interface.");
            end
        end 

        push!(rscones,rscone)
    end 

    return rscones
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
