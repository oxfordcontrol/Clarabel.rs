#![allow(non_snake_case)]
use crate::cones::*;
use crate::algebra::*;
use crate::settings::*;
use crate::kktsolvers::KKTSolver;
//use crate::kktsolvers::direct_ldl::*; //PJG:This is a horrendous include
use crate::kktsolvers::direct_quasidefinite::utils::*;
use crate::kktsolvers::direct_quasidefinite::datamap::*;
use crate::kktsolvers::direct_quasidefinite::DirectLDLSolver;
use crate::kktsolvers::direct_quasidefinite::ldlsolvers::QDLDLDirectLDLSolver;

// -------------------------------------
// KKTSolver using direct LDL factorisation
// -------------------------------------

pub struct DirectQuasidefiniteKKTSolver<'a, T: FloatT = f64>
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

    // a vector for storing the entries of WtW blocks
    // ons the KKT matrix block diagonal
    WtWblocks: Vec<Vec<T>>,

    // settings just points back to the main solver settings.
    // Required since there is no separate LDL settings container
    settings: &'a Settings<T>,

    // the direct linear LDL solver
    ldlsolver: Box<dyn DirectLDLSolver<T>>,
}

//PJG: check on lifetime reqs here (actually everywhere in this file)
impl<'a, T: FloatT> DirectQuasidefiniteKKTSolver<'a, T> {
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
            let mut signs = vec![1_i8; n+m+p];
            _fill_signs(&mut signs, m, n, p);

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
            let (KKT, map) = _assemble_kkt_matrix(P,A,cones,kktshape);

            //PJG: switchable solver engine not implemented yet
            // the LDL linear solver engine
            let mut ldlsolver = Box::new(QDLDLDirectLDLSolver::<T>::new(KKT,signs,settings));

            if settings.static_regularization_enable {
                let eps = settings.static_regularization_eps;
                ldlsolver.offset_values(&map.diagP,eps);
            }

            Self{
                m,n,p,x,b,map,WtWblocks,settings,ldlsolver,
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

fn _fill_signs(signs: &mut [i8], m: usize,n: usize,p: usize){

    signs.fill(1);

    //flip expected negative signs of D in LDL
    signs[n..(n+m)]
        .iter_mut()
        .for_each(|x| *x = -*x);

    //the trailing block of p entries should
    //have alternating signs
    signs[(n+m)..(n+m+p)]
        .iter_mut()
        .step_by(2)
        .for_each(|x| *x = -*x);
}



impl<'a, T> KKTSolver<'a, T> for DirectQuasidefiniteKKTSolver<'a, T>
where
    T: FloatT,
{
    fn update(&mut self, cones: ConeSet<T>){

        let settings  = &self.settings;
        let map       = &self.map;
        let ldlsolver = &mut self.ldlsolver;


        // Set the elements the W^tW blocks in the KKT matrix.
        cones.get_WtW_block(&mut self.WtWblocks);

        for (values, index) in self.WtWblocks.iter_mut().zip(map.WtWblocks.iter()){
            // change signs to get -W^TW
            values.negate();
            ldlsolver.update_values(index,values);
        }

        // update the scaled u and v columns.
        let mut cidx = 0;        // which of the SOCs are we working on?
        for (i,_cone) in cones.iter().enumerate(){

            if cones.types[i] == SupportedCones::SecondOrderConeT {

                //here we need to recover the inner SOC value for
                //this cone so we can access its fields

                //PJG: This is extremely questionable
                let K = cones.anyref_by_idx(i);
                let K = K.downcast_ref::<SecondOrderCone<T>>();

                match K {
                    None => {panic!("cone type list is corrupt.");}
                    Some(K) => {


                        let η2 = T::powi(K.η,2);

                        //off diagonal columns (or rows)
                        //PJG: not sure how to force the scaling here
                        //commenting out for the moment
                        //ldlsolver.update_values(&map.SOC_u[cidx],(-η2).*K.u);
                        //ldlsolver.update_values(&map.SOC_v[cidx],(-η2).*K.v);

                        //add η^2*(1/-1) to diagonal in the extended rows/cols
                        ldlsolver.update_values(&map.SOC_D[cidx*2..],&[-η2;1]);
                        ldlsolver.update_values(&map.SOC_D[cidx*2..],&[ η2;1]);

                        cidx += 1;
                    }
                } //end match
            } //end if SOC
        } //end for

        // Perturb the diagonal terms WtW that we have just overwritten
        // with static regularizers.  Note that we don't want to shift
        // elements in the ULHS #(corresponding to P) since we already
        // shifted them at initialization and haven't overwritten it
        if settings.static_regularization_enable {
            let eps = settings.static_regularization_eps;
            ldlsolver.offset_values(&map.diag_full,eps);
            ldlsolver.offset_values(&map.diagP,-eps);  //undo to the P shift
        }

        //refactor with new data
        ldlsolver.refactor();

    } //end fn


    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]){

        let (m,n,p) = (self.m,self.n,self.p);

        self.b[0..n].copy_from(rhsx);
        self.b[n..(n+m)].copy_from(rhsz);
        self.b[n+m..(n+m+p)].fill(T::zero());

    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>
    ){

        self.ldlsolver.solve(&mut self.x, &self.b);
        self.getlhs(lhsx,lhsz);

    }
}


// extra helper functions, not required for KKTSolver trait

impl<'a, T:FloatT> DirectQuasidefiniteKKTSolver<'a, T> {
    fn getlhs(
        &self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>
    ) {

        let x = &self.x;
        let (m,n,_p) = (self.m,self.n,self.p);

        if let Some(v) = lhsx {
            v.copy_from(&x[0..n]);
        }
        if let Some(v) = lhsz {
            v.copy_from( &x[n..(n+m)]);
        }

    }
}
