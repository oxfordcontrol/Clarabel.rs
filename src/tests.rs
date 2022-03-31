#[cfg(test)]
mod tests {
    #[test]
    fn dim_numel_degree() {
        let zcone = crate::cones::ZeroCone::new(5);
        assert_eq!(zcone.dim(), 5);
        assert_eq!(zcone.numel(), 5);
        assert_eq!(zcone.degree(), 0);
    }

    #[test]
    fn rectify_equilibration() {
        let zcone = crate::cones::ZeroCone::new(5);
        let e = [1., 2., 3.];
        let mut δ = [0., 0., 0.];
        zcone.rectify_equilibration(&mut δ, &e);
        assert_eq!(δ, [1., 2., 3.]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn get_WtW_block() {
        let zcone = crate::cones::ZeroCone::new(5);
        let mut w = [1., 2., 3.];
        zcone.get_WtW_block(&mut w);
        assert_eq!(w, [0., 0., 0.]);
    }

    #[test]
    fn λ_circ_λ() {
        let zcone = crate::cones::ZeroCone::new(5);
        let mut x = [1., 2., 3.];
        zcone.λ_circ_λ(&mut x);
        assert_eq!(x, [0., 0., 0.]);
    }
}
