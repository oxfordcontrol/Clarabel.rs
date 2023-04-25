use crate::solver::core::cones::*;

#[test]
fn numel_degree() {
    let zcone = ZeroCone::<f64>::new(5);
    let nncone = NonnegativeCone::<f64>::new(5);
    let scone = SecondOrderCone::<f64>::new(5);
    let expcone = ExponentialCone::<f64>::new();
    let powcone = PowerCone::<f64>::new(0.5);

    assert_eq!(zcone.numel(), 5);
    assert_eq!(zcone.degree(), 0);
    assert_eq!(nncone.numel(), 5);
    assert_eq!(nncone.degree(), 5);
    assert_eq!(scone.numel(), 5);
    assert_eq!(scone.degree(), 1);
    assert_eq!(expcone.numel(), 3);
    assert_eq!(expcone.degree(), 3);
    assert_eq!(powcone.numel(), 3);
    assert_eq!(powcone.degree(), 3);

    #[cfg(feature = "sdp")]
    {
        let sdpcone = PSDTriangleCone::<f64>::new(5);
        assert_eq!(sdpcone.numel(), 15);
        assert_eq!(sdpcone.degree(), 5);
    }
}
