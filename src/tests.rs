#[allow(unused_imports)]
#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    fn dim_numel_degree() {
        let zcone = ZeroCone::<f64>::new(5);
        let nncone = NonnegativeCone::<f64>::new(5);
        assert_eq!(zcone.dim(), 5);
        assert_eq!(zcone.numel(), 5);
        assert_eq!(zcone.degree(), 0);
        assert_eq!(nncone.dim(), 5);
        assert_eq!(nncone.numel(), 5);
        assert_eq!(nncone.degree(), 5);
    }
    //
    #[test]
    fn rectify_equilibration_slice() {
        let zcone = ZeroCone::<f64>::new(5);
        let e = vec![1., 2., 3.];
        let mut δ = vec![0., 0., 0.];
        zcone.rectify_equilibration(&mut δ, &e);
        assert_eq!(δ, vec![1., 2., 3.]);
    }

    #[test]
    fn rectify_equilibration() {
        let zcone = ZeroCone::<f64>::new(5);
        let e = vec![1., 2., 3.];
        let es = &e[0..3];
        let mut δ = vec![0., 0., 0.];
        zcone.rectify_equilibration(&mut δ, &es);
        assert_eq!(δ, vec![1., 2., 3.]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn get_WtW_block() {
        let zcone = ZeroCone::<f64>::new(5);
        let mut w = vec![1., 2., 3.];
        zcone.get_WtW_block(&mut w);
        assert_eq!(w, vec![0., 0., 0.]);
    }

    #[test]
    fn λ_circ_λ() {
        let zcone = ZeroCone::<f64>::new(5);
        let mut x = vec![1., 2., 3.];
        zcone.λ_circ_λ(&mut x);
        assert_eq!(x, vec![0., 0., 0.]);
    }
}
