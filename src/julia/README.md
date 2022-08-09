This directory provides a Julia wrapper for the Rust implementation of Clarabel.   
It provides a Julia package called "ClarabelRs", which can be used in place of 
the equivalent pure Julia package "Clarabel".  The main "Clarabel" package in 
Julia provides a shared MOI wrapper for these solvers, so JuMP works with both.

This package is intended only for development and benchmarking purposes by the  
Clarabel developers.   It may be modified / withdrawn / broken without warning.

If you just want to use the Clarabel solver in Julia, you are advised to use 
the native Julia implementation here:  https://github.com/oxfordcontrol/Clarabel.jl

You can also install the native Julia version directly using: 
julia> ]add Clarabel 


