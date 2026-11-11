use crate::device::{self, Device};

#[test]
fn canonical_and_legacy_device_paths_share_type_policy_and_checkpoint_format() {
    let legacy: crate::autograd::Device = Device::Cuda;
    assert_eq!(legacy, Device::Cuda);
    for (value, variant, name) in [
        (Device::Cpu, 0u32, "\"Cpu\""),
        (Device::Cuda, 1, "\"Cuda\""),
    ] {
        assert_eq!(bincode::serialize(&value).unwrap(), variant.to_le_bytes());
        assert_eq!(serde_json::to_string(&value).unwrap(), name);
    }

    let original = device::is_strict_device_execution();
    let _restore = device::set_strict_device_execution_scoped(original);
    crate::autograd::set_strict_device_execution(false);
    assert!(!device::is_strict_device_execution());
    {
        let _outer = crate::autograd::set_strict_device_execution_scoped(true);
        assert!(device::is_strict_device_execution());
        {
            let _inner = device::set_strict_device_execution_scoped(false);
            assert!(!crate::autograd::is_strict_device_execution());
        }
        assert!(crate::autograd::is_strict_device_execution());
    }
    assert!(!device::is_strict_device_execution());

    let panic = std::panic::catch_unwind(|| {
        let _guard = crate::autograd::set_strict_device_execution_scoped(true);
        panic!("test scoped device restoration");
    });
    assert!(panic.is_err());
    assert!(!device::is_strict_device_execution());
    std::thread::spawn(|| {
        let _guard = device::set_strict_device_execution_scoped(true);
        assert!(crate::autograd::is_strict_device_execution());
    })
    .join()
    .unwrap();
    assert!(!crate::autograd::is_strict_device_execution());
}
