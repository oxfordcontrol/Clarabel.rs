
To do:

Check code against no_std

I have defined CompositeCone as a core cone type, and use this directly in the default implementation.  Maybe it should be DefaultCone === CompositeCone in that implementation though.   It also seems like the functions that populate the WtW block in the CompositeCone are implementation specific.   If so, these should be pulled out into DefaultCone.  Note Julia file structure and method names are no longer the same.

Take only upper triangle of P in DefaultProblemData.   This means that something better than clone is required in the constructor.

There are no checks at all on CSC constructor.   At least nzval and rowval should be the same length, and colptr should agree with dims.w

Use of underscores in private function names is totally inconsistent.   What is the usual style, e.g. for the rust float formatting internal functions?

Change SolveResult to Solution and then SolveInfo back to Info.

Remove FloatT strait bounds from struct definitions and keep in the impl only?   It is also unclear whether it is really
adding value to put a generic f64 on all of these definitions.

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

The CscMatrix is defined as part of algebra crate, which means that both algebra and solver crates need to be separately imported in my python example.  It's not clear where the CscMatrix definition should live.   Maybe algebra shouldn't be a separate crate, or  the whole collection of crates needs to be reexported somehow.

Remove lifetime annotation on KKTsolvers if they are still there.  At the moment I think KKTsolver object
is taking a copy of the settings to support iterative refinement.

Consider whether SolveResult can be made a dependency only in the same way as Cone in the top level solver.   Maybe not since it depends on descaling equilibration stuff.

Consider whether Settings can be made dependency only as well.   This should be easier since there are minimal direct field accesses in the top level solver.  Probably only max_step_fraction() is needed as a getter in solver to allow a generic settings input by trait there.

I am currently disassembling the cone index ranges in the KKT assembly function to pass the headidx to the different colcount and colfill functions, and then building the ranges again there.   Maybe could be improved.   [This might be fixed already?]


kkt_fill and friends are taking a length index as the final argument, but this seems redundant.   Maybe it was there to facilitate a C version.

Really confusing native methods implementation for CSC matrices.   What should nalgebra and others implement?   Should this go somewhere else?   Maybe I just need a "SparseMatrix" or "Matrix" trait.

Settings uses time_limit in Rust and max_time in Julia.   Or maybe time_limit is a bool   Very confusing in the settings print function.   Maybe this was just a bug in the no time limit case?   Fixed in Julia print maybe.   Perhaps Rust should use Option here.  


maybe _offset_diagonal_KKT should be a method on a sparse matrix, rather than something implemented with the KKT solver code.   Could also add something like _assign_diagonal_KKT.   

SupportedCones is really part of the default implementation's API, and not really part of the Cone trait that defines all of the required behaviours.   These should be separated I think.

I want to remove trait bounds on struct definitions, but I can't do so completely because of this: 
    struct Settings<T: FloatT>
Settings must have FloatT as a bound so that Builder will work.   But then I must impose the same trait bound and everyhing that includes Settings as a field (and so on, recursively down).  This is a problem specifically with the KKTSolver that takes a copied Settings as an argument, which means that things like the QDLDL solver and associated sctructures are also getting this trait bound.

Maybe the Settings in the KKTSolver should be a borrow with a lifetime, but I don't know how to implement this.

The Ruiz equilibration is a method implemented in DefaultProblemData.  Maybe it should be associated with the DefaultEquilibration type instead.

Julia compat updates:
---------------------

Removed data as an argument to aff/combined RHS calcs.   Removed a few other unused params for other top level functions (settings?   Should have written it down.)

Caution: in QDLDL, changing the Dsigns to be the internally permuted ones means that the offset_values function will need to change.   This was a Rust bug, now fixed (??) in the rust impl.   Much of this code be avoided if we only update one version of the KKT matrix and then just memcpy it through the permutation on a refactor.

Residuals and Variables should take (m,n) sizes.   There is no consistency about whether n comes first or m in argument lists.   This is super confusing.

I took cones out of the argument list for the variables constructor because I am no longer using ConicVector.  Now it just takes (m,n);

Change QDLDL dsign behaviour to be consistent with Rust.   QDLDL should use internally held signs when applying offsets, not an externally supplied vector.

Changed cone_scaling_update call in top level solver to a method scale_cones on the variables.   This way only Variables/Residuals/Data/KKT/SolveInfo/SolveResult need to be mutually interoperable, since the Cone type has no methods taking any of these as arguments.

Possible removal of ConicVector and switch to range calcs as in Rust.   This might be slightly slower.  REmember to @view on the ranges.

Solver is failing now with Cholmod or MKL.   I don't understand why.   Possibly related to updated Dsigns behaviour or to the new scale_update calls.  Maybe I fixed it already?

I moved the timers into the main solver struct and out of info.  The info struct is user implemented and shouldn't be expected
to provide timing facilities for main solver internals.

Final few steps of main solver loop are slightly reordered to facilitate printing etc.   Go back to Julia and make it the same sequence.
