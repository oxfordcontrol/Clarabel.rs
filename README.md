
To do:

Remove trait bounds from struct definitions and keep in the impl only.

Inconsistent usage between :
  impl<T> ... where T: FloatT,
and
  impl<T:FloatT>
Pick a lane here.


Move IR implementation to within QDLDL.   Move all IR settings to within QDLDL and then remove lifetime annotation on KKTsolver that are supporting shared references to the master settings object.

Apply builder method to master settings and implement defaults that way instead.

Consider whether SolveResult can be made a dependency only in the same way as Cone in the top level solver.   Maybe not since it depends on descaling equilibration stuff.

Consider whether Settings can be made dependency only as well.   This should be easier since there are minimal direct field accesses in the top level solver.  Probably only max_step_fraction() is needed as a getter in solver to allow a generic settings input by trait there.

I am currently disassemble the cone index ranges in the KKT assembly function to pass the headidx to the different colcount and colfill functions, and then building the ranges again there.   Maybe could be improved.

Remove ConicVector as a mod and from all use statements.

Maybe ConeSet<T> Should be DefaultCone<T> and should be placed into the default implementation group.


conicvector.rs should now be deleted.

Lifetime in DirectQuasidefiniteKKTSolver should be removed once settings are rectified.




Julia compat updates:

Change QDLDL dsign behaviour to be consistent with Rust.   QDLDL should use internally held signs when applying offsets, not an externally supplied vector.

Changed cone_scaling_update call in top level solver to a method scale_cones on the variables.   This way only Variables/Residuals/Data/KKT/SolveInfo/SolveResult need to be mutually interoperable, since the Cone type has no methods taking any of these as arguments.

Possible removal of ConicVector and switch to range calcs as in Rust.   This might be slightly slower.  REmember to @view on the ranges.
