use crate::algebra::*;

#[test]
fn test_copy_from() {
    let x = vec![3., 0., 2., 1.];
    let mut y = vec![0.; 4];
    y.copy_from(&x);
    assert_eq!(x, y);
}

#[test]
fn test_select() {
    let x = [1., 2., 3., 4., 5.];
    let idx = [true, false, true, true, false];
    let y = x.select(&idx);
    assert_eq!(y, vec![1., 3., 4.]);
}

#[test]
fn test_scalarop() {
    let mut x = vec![3., 0., 2., 1.];
    x.scalarop(|x| -2. * x);
    assert_eq!(x, vec![-6., 0., -4., -2.]);
}

#[test]
fn test_scalarop_from() {
    let x: Vec<f64> = vec![3., 0., 2., 1.];
    let mut y: Vec<f64> = vec![0.; 4];
    y.scalarop_from(|x| -2. * x, &x);
    assert_eq!(y, vec![-6., 0., -4., -2.]);
}

#[test]
fn test_translate() {
    let mut x = [3., 0., 2., 1.];
    x.translate(-4.);
    assert_eq!(x, [-1., -4., -2., -3.]);
}

#[test]
fn test_scale() {
    let mut x = [3., 0., 2., 1.];
    x.scale(3.);
    assert_eq!(x, [9., 0., 6., 3.]);
}

#[test]
fn test_recip() {
    let mut x = [3., 10., 2., 1.];
    x.recip();
    assert!(x.norm_inf_diff(&[1. / 3., 1. / 10., 1. / 2., 1.]) < 1e-8);
}

#[test]
fn test_sqrt() {
    let mut x = vec![9., 4., 16., 0.];
    x.sqrt();
    assert_eq!(x, vec![3., 2., 4., 0.]);
}

#[test]
fn test_rsqrt() {
    let mut x = vec![9., 4., 16., 1.];
    x.rsqrt();
    assert_eq!(x, vec![1. / 3., 1. / 2., 1. / 4., 1.]);
}

#[test]
fn test_negate() {
    let mut x = vec![9., 4., 16., 1.];
    x.negate();
    assert_eq!(x, vec![-9., -4., -16., -1.]);
}

#[test]
fn test_hadamard() {
    let mut x = vec![1., 2., 3., 4.];
    let s = vec![-1., -2., -4., 8.];
    x.hadamard(&s);
    assert_eq!(x, vec![-1., -4., -12., 32.]);
}

#[test]
fn test_clip() {
    let min_thresh = 0.1;
    let max_thresh = 10.;
    let mut x = vec![0.01, 0.1, 1., 10., 100.];
    x.clip(min_thresh, max_thresh);

    assert_eq!(x, vec![0.1, 0.1, 1., 10., 10.]);
}

#[test]
fn test_op_chaining() {
    let x = vec![5., 1., 3., 7.];
    let mut y = vec![1.; 4];
    y.axpby(1., &x, 3.).recip().hadamard(&[1., 2., 3., 4.]);
    assert_eq!(y, vec![0.125, 0.5, 0.5, 0.4]);
}

#[test]
fn test_dot() {
    let x = vec![3., 0., 2., 1.];
    let y = vec![-1., -2., 3., 4.];

    assert_eq!(x.dot(&y), 7.);
    assert_eq!(y.dot(&x), 7.);
}

#[test]
fn test_dist() {
    let x = vec![3., 0., 2., 1.];
    let y = vec![-1., -2., 3., 4.];

    assert_eq!(x.dist(&y), f64::sqrt(30.));
    assert_eq!(y.dist(&x), f64::sqrt(30.));
}

#[test]
fn test_sumsq() {
    let x = [-1., 2., -3., 4.];
    assert_eq!(x.sumsq(), 30.);
}

#[test]
fn test_norm() {
    let x = [-3., 4., -12.];
    assert_eq!(x.norm(), 13.);
}

#[test]
fn test_norm_scaled() {
    let x = [-3. / 2., 4. / 3., -12. / 4.];
    let s = [2., 3., 4.];
    assert_eq!(x.norm_scaled(&s), 13.);
}

#[test]
fn test_norm_inf() {
    let x = [-3., 4., -12.];
    assert_eq!(x.norm_inf(), 12.);

    let x = [-3., f64::NAN, -12.];
    assert!(x.norm_inf().is_nan());
}

#[test]
fn test_norm_one() {
    let x = [-3., 4., -12.];
    assert_eq!(x.norm_one(), 19.);
}

#[test]
fn test_minimum() {
    let x = [-3., 4., -12.];
    assert_eq!(x.minimum(), -12.);
}

#[test]
fn test_maximum() {
    let x = [-3., 4., -12.];
    assert_eq!(x.maximum(), 4.);
}

#[test]
fn test_mean() {
    let x = [-3., 4., -12., -1.];
    assert_eq!(x.mean(), -3.);
}

#[test]
fn test_axpby() {
    let x = vec![3., 0., 2., 1.];
    let mut y = vec![-1., -2., -1., 0.];
    let a = 2.;
    let b = 3.;

    //y = ax + by
    y.axpby(a, &x, b);

    assert_eq!(y, [3., -6., 1., 2.]);
}

#[test]
fn test_waxpby() {
    let x = vec![3., 0., 2., 1.];
    let y = vec![-1., -2., -1., 0.];
    let a = 2.;
    let b = 3.;
    let mut w = vec![0f64; 4];

    //w = ax + by
    w.waxpby(a, &x, b, &y);

    assert_eq!(w, [3., -6., 1., 2.]);
}
