use super::*;

#[test]
fn offline_quantization_boundary_handles_tiny_manual_scale() {
    let scale = f32::from_bits(0x10000);
    let input = [0.0, scale, -scale, 2.0 * scale];
    assert_eq!(
        quantize_f32_slice_to_i8(&input, Some(scale)).unwrap().0,
        vec![0, 1, -1, 2]
    );
    let brain = input.map(bf16::from_f32);
    assert_eq!(
        quantize_bf16_slice_to_i8(&brain, Some(scale)).unwrap().0,
        vec![0, 1, -1, 2]
    );
    let half = input.map(f16::from_f32);
    assert_eq!(
        quantize_f16_slice_to_i8(&half, Some(scale)).unwrap().0,
        vec![0; 4]
    );
}

#[test]
fn offline_i8_quantization_rejects_non_finite_values() {
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(quantize_f32_slice_to_i8(&[value], None).is_err());
        assert!(quantize_f16_slice_to_i8(&[f16::from_f32(value)], None).is_err());
        assert!(quantize_bf16_slice_to_i8(&[bf16::from_f32(value)], None).is_err());
    }
}

#[test]
fn offline_i8_quantization_rejects_invalid_manual_scale() {
    for scale in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert!(quantize_f32_slice_to_i8(&[1.0], Some(scale)).is_err());
    }
}

#[test]
fn offline_i8_quantization_keeps_subnormal_scale_positive() {
    let (_, scale) =
        quantize_f32_slice_to_i8(&[f32::from_bits(1)], None).expect("quantize subnormal");
    assert_eq!(scale, f32::MIN_POSITIVE);
}

#[test]
fn existing_i8_scale_name_is_preserved() {
    let mut i8_names = HashSet::new();
    i8_names.insert("layer.weight".to_string());
    assert!(is_existing_i8_scale_name("layer.weight.scale", &i8_names));
    assert!(!is_existing_i8_scale_name("layer.bias.scale", &i8_names));
    assert!(!is_existing_i8_scale_name("layer.weight", &i8_names));
}
