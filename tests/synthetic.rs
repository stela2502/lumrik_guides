use lumrik_guides::stats::{posterior_real, poisson_upper_tail};

#[test]
fn genuine_guide_can_be_called_without_winner_take_all() {
    // Two independently strong guides in the same hypothetical cell.
    let a = posterior_real(50, 0.2, 0.05, 35.0, 10.0);
    let b = posterior_real(42, 0.1, 0.05, 30.0, 10.0);

    assert!(a > 0.95);
    assert!(b > 0.95);
}

#[test]
fn guide_specific_ambient_frequency_matters() {
    // Same observed UMI count, very different ambient expectations.
    let rare_guide = poisson_upper_tail(4, 0.05);
    let common_guide = poisson_upper_tail(4, 3.0);

    assert!(rare_guide < common_guide);
}
