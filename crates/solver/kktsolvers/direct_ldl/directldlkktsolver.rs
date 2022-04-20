use crate::cones::*;
use crate::algebra::*;
use crate::settings::*;
use crate::kktsolvers::KKTSolver;
//use crate::kktsolvers::direct_ldl::*; //PJG:This is a horrendous include
use crate::kktsolvers::direct_ldl::utils::*;
use crate::kktsolvers::direct_ldl::ldldatamap::*;

// -------------------------------------
// KKTSolver using direct LDL factorisation
// -------------------------------------

pub struct DirectLDLKKTSolver<'a, T: FloatT = f64>
{
    // problem dimensions
    m: usize,
    n: usize,
    p: usize,

    // Left and right hand sides for solves
    x: Vec<T>,
    b: Vec<T>,

    // KKT mapping from problem data to KKT
    map: LDLDataMap,

    // the expected signs of D in KKT = LDL^T
    Dsigns: Vec<i8>,

    // a vector for storing the entries of WtW blocks
    // ons the KKT matrix block diagonal
    WtWblocks: Vec<Vec<T>>,

    // settings just points back to the main solver settings.
    // Required since there is no separate LDL settings container
    settings: &'a Settings<T>,

    // the direct linear LDL solver
    ldlsolver: usize, //Box<dyn impl DirectLDLSolver<T>>,
}

impl<'a, T: FloatT> DirectLDLKKTSolver<'a, T> {
    pub fn new(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &ConeSet<T>,
        m: usize,
        n: usize,
        settings: &'a Settings<T>) -> Self
        {
            // solving in sparse format.  Need this many
            // extra variables for SOCs
            let p = 2*cones.type_counts[&SupportedCones::SecondOrderConeT];

            // LHS/RHS/work for iterative refinement
            let x    = vec![T::zero();n+m+p];
            let b    = vec![T::zero();n+m+p];

            // the expected signs of D in LDL
            let mut Dsigns = vec![1 as i8; n+m+p];
            _fill_Dsigns(&mut Dsigns, m, n, p);

            // updates to the diagonal of KKT will be
            // assigned here before updating matrix entries
            let WtWblocks = _allocate_kkt_WtW_blocks::<T,T>(cones);

            // which LDL solver should I use?
            //PJG: commented out and QDLDL harcodded for now
            // ldlsolverT = _get_ldlsolver_type(settings.direct_solve_method);

            //PJG: hardcoding shape
            // does it want a :triu or :tril KKT matrix?
            //kktshape = required_matrix_shape(ldlsolverT);
            let kktshape = MatrixTriangle::Triu;
            let (KKT, map) = _assemble_kkt_matrix(&P,&A,cones,kktshape);

            //PJG: solver engine not implemented yet
            // the LDL linear solver engine
            let ldlsolver = 0; //= ldlsolverT{T}(KKT,Dsigns,settings);

            if settings.static_regularization_enable {
                let eps = settings.static_regularization_eps;
                ldlsolver.offset_values(&map.diagP,eps);
            }

            Self{
                m: m,
                n: n,
                p: p,
                x: x,
                b: b,
                map: map,
                Dsigns: Dsigns,
                WtWblocks: WtWblocks,
                settings:  settings,
                ldlsolver: ldlsolver,
            }
        }
    }


//PJG: Switching not supported yet.   Fix this.
// function _get_ldlsolver_type(s::Symbol)
//     try
//         return DirectLDLSolversDict[s]
//     catch
//         throw(error("Unsupported direct LDL linear solver :", s))
//     end
// end

fn _fill_Dsigns(Dsigns: &[i8], m: usize,n: usize,p: usize){

    Dsigns.fill(0 as i8);

    //flip expected negative signs of D in LDL
    Dsigns[n..(n+m)]
        .iter_mut()
        .for_each(|x| *x = -*x);

    //the trailing block of p entries should
    //have alternating signs
    Dsigns[(n+m)..(n+m+p)]
        .iter_mut()
        .step_by(2)
        .for_each(|x| *x = -*x);
}



impl<'a, T> KKTSolver<'a, T> for DirectLDLKKTSolver<'a, T>
where
    T: FloatT,
{
    fn update(&mut self, cones: ConeSet<T>){

        let (m,n,p) = (self.m,self.n,self.p);

        let settings  = self.settings;
        let map       = self.map;


        // Set the elements the W^tW blocks in the KKT matrix.
        cones.get_WtW_block(&mut self.WtWblocks);

        for (index, values) in map.WtWblocks.iter().zip(self.WtWblocks){
            // change signs to get -W^TW
            values.negate();
            self.update_values(index,values);
        }

        // update the scaled u and v columns.
        let mut cidx = 0;        // which of the SOCs are we working on?
        for (i,cone) in cones.iter().enumerate(){
            if cones.types[i] == SupportedCones::SecondOrderConeT {

                    let η2 = cone.η^2;

                    //off diagonal columns (or rows)
                    //PJG: not sure how to force the scaling here
                    //commenting out for the moment
                    //ldlsolver.update_values(&map.SOC_u[cidx],(-η2).*K.u);
                    //ldlsolver.update_values(&map.SOC_v[cidx],(-η2).*K.v);

                    //add η^2*(1/-1) to diagonal in the extended rows/cols
                    self.update_values(&map.SOC_D[cidx*2..],&vec![-η2;1]);
                    self.update_values(&map.SOC_D[cidx*2..],&vec![η2;1]);

                    cidx += 1;
            }

        }

        // Perturb the diagonal terms WtW that we have just overwritten
        // with static regularizers.  Note that we don't want to shift
        // elements in the ULHS #(corresponding to P) since we already
        // shifted them at initialization and haven't overwritten it
        if settings.static_regularization_enable {
            let eps = settings.static_regularization_eps;
            self.offset_values(&map.diag_full,eps,&self.Dsigns);
            self.offset_values(&map.diagP,&vec![-eps;1]);  //undo to the P shift
        }

        //refactor with new data
        self.ldlsolver.refactor();

    }


    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]){

        let (m,n,p) = (self.m,self.n,self.p);

        self.b[0..n].copy_from(&rhsx);
        self.b[n..(n+m)].copy_from(&rhsz);
        self.b[n+m..(n+m+p)].fill(T::zero());

    }

    fn solve(
        &self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>
    ){

        let (x,b) = (self.x,self.b);
        self.ldlsolver.solve(&x,&b,&self.settings);
        self.getlhs(lhsx,lhsz);

    }
}


// extra helper functions, not required for KKTSolver trait

impl<'a, T:FloatT> DirectLDLKKTSolver<'a, T> {
    fn getlhs(
        &self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>
    ) {

        let x = &self.x;
        let (m,n,p) = (self.m,self.n,self.p);

        if let Some(v) = lhsx {
            v.copy_from(&x[0..n]);
        }
        if let Some(v) = lhsz {
            v.copy_from( &x[n..(n+m)]);
        }

    }
}
