use super::{InitType, ParameterInitMode, tensor_init, with_parameter_init_mode};
use crate::precision::{DType, PrecisionConfig, with_precision_config};

#[test]
fn placeholder_parameter_init_overrides_random_initializers() {
    let tensor = with_parameter_init_mode(ParameterInitMode::Placeholder, || {
        tensor_init(vec![2, 3], InitType::KaimingNormal)
    });
    assert!(tensor.data_ref().iter().all(|&v| v == 0.0));
}

#[test]
fn placeholder_parameter_init_follows_global_parameter_dtype() {
    with_precision_config(
        PrecisionConfig {
            parameter_dtype: DType::I8,
            runtime_dtype: DType::F32,
            allow_parameter_dtype_copies: false,
        },
        || {
            let tensor = with_parameter_init_mode(ParameterInitMode::Placeholder, || {
                tensor_init(vec![2, 3], InitType::KaimingNormal)
            });
            assert_eq!(tensor.dtype(), DType::I8);
        },
    );
}
