
To do:

Check code against no_std

There are no checks at all on CSC constructor.   At least nzval and rowval should be the same length, and colptr should agree with dims.w

Use of underscores in private function names is totally inconsistent.   What is the usual style, e.g. for the rust float formatting internal functions?

Check basic QP equilibration values (step through code)

Change SolveResult to Solution and then SolveInfo back to Info.

Remove trait bounds from struct definitions and keep in the impl only.

Printing functions should be a totally separate trait on DefaultSolveInfo, and should
be separately implementable, or at the very least in a separate file.  Solver should require that both are implemented, then disabling printing amounts to declaring the trait to be empty or similar.

T::from(...).unwrap() sucks in settings and elsewhere.   Can I use into()?

Add some kind of T::const() method to FloatT to aid conversions like 2, 0.5 etc.
The current unwrap business is dreadful.  I fee like this is what from should do, but I don't think (??) it works properly.  Maybe only need two() and half().

Inconsistent usage between :
  impl<T> ... where T: FloatT,
and
  impl<T:FloatT>
Pick a lane here.


Move IR implementation to within QDLDL.   Move all IR settings to within QDLDL and then remove lifetime annotation on KKTsolver that are supporting shared references to the master settings object.

Consider whether SolveResult can be made a dependency only in the same way as Cone in the top level solver.   Maybe not since it depends on descaling equilibration stuff.

Consider whether Settings can be made dependency only as well.   This should be easier since there are minimal direct field accesses in the top level solver.  Probably only max_step_fraction() is needed as a getter in solver to allow a generic settings input by trait there.

I am currently disassembling the cone index ranges in the KKT assembly function to pass the headidx to the different colcount and colfill functions, and then building the ranges again there.   Maybe could be improved.   [This might be fixed already?]

Remove ConicVector as a mod and from all use statements.  [Might be fixed already]

Maybe ConeSet<T> Should be DefaultCone<T> and should be placed into the default implementation group.   Problem here because Julia uses slightly different method names.

conicvector.rs should now be deleted.

kkt_fill and friends are taking a length index as the final argument, but this seems redundant.   Maybe it was there to facilitate a C version.

Lifetime in DirectQuasidefiniteKKTSolver should be removed once settings are rectified.  [I think this is done already]

Really confusing native methods implementation for CSC matrices.   What should nalgebra and others implement?   Should this go somewhere else?

Change symdot to quad_form.   Do this also in Julia.

Don't use T::recip(T::epsilon()), but rather T::max_value() of T::infinity() as appropriate.  Same for Julia probably.

Settings uses time_limit in Rust and max_time in Julia.   Or maybe time_limit is a bool   Very confusing in the settings print function.   Maybe this was just a bug in the no time limit case?   Fixed in Julia print maybe.   Perhaps Rust should use Option here.  

Julia compat updates:

Removed data as an argument to aff/combined RHS calcs.   Removed a few other unused params for other top level functions (settings?   Should have written it down.)

Caution: in QDLDL, changing the Dsigns to be the internally permuted ones means that the offset_values function will need to change.   This was a Rust bug, now fixed (??) in the rust impl.

Residuals and Variables should take (m,n) sizes.   There is no consistency about whether n comes first or m in argument lists.   This is super confusing.

I took cones out of the argument list for the variables constructor because I am no longer using ConicVector.  Now it just takes (m,n);

Change QDLDL dsign behaviour to be consistent with Rust.   QDLDL should use internally held signs when applying offsets, not an externally supplied vector.

Changed cone_scaling_update call in top level solver to a method scale_cones on the variables.   This way only Variables/Residuals/Data/KKT/SolveInfo/SolveResult need to be mutually interoperable, since the Cone type has no methods taking any of these as arguments.

Possible removal of ConicVector and switch to range calcs as in Rust.   This might be slightly slower.  REmember to @view on the ranges.
