use super::*;

#[test]
fn collection_restores_nested_scopes_and_phases() {
    let ((), regions) = capture(|| {
        let _phase = phase("decode");
        measure("outer", || {
            let (value, inner) = capture(|| {
                let _phase = phase("prefill");
                measure("inner", || 42)
            });
            assert_eq!(value, 42);
            assert_eq!(inner[0].phase, "prefill");
        });
        measure("outer", || ());
    });
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].phase, "decode");
    assert_eq!(regions[0].calls, 2);
    assert!(ACTIVE.with(|s| s.borrow().is_none()));
}

#[test]
fn unwind_restores_previous_collection() {
    let (_, regions) = capture(|| {
        let _phase = phase("decode");
        let result = std::panic::catch_unwind(|| capture(|| panic!("probe")));
        assert!(result.is_err());
        measure("after", || ());
    });
    assert_eq!(regions[0].stage, "after");
    assert_eq!(regions[0].phase, "decode");
}
