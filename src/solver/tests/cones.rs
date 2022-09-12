use crate::solver::core::cones::*;

#[test]
fn dim_numel_degree() {
    let zcone = ZeroCone::<f64>::new(5);
    let nncone = NonnegativeCone::<f64>::new(5);
    let scone = SecondOrderCone::<f64>::new(5);
    let expcone = ExponentialCone::<f64>::new();
    assert_eq!(zcone.dim(), 5);
    assert_eq!(zcone.numel(), 5);
    assert_eq!(zcone.degree(), 0);
    assert_eq!(nncone.dim(), 5);
    assert_eq!(nncone.numel(), 5);
    assert_eq!(nncone.degree(), 5);
    assert_eq!(scone.dim(), 5);
    assert_eq!(scone.numel(), 5);
    assert_eq!(scone.degree(), 1);
    assert_eq!(expcone.dim(), 3);
    assert_eq!(expcone.numel(), 3);
    assert_eq!(expcone.degree(), 3);
}
