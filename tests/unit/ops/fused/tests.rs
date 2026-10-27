use super::*;
use crate::autograd::no_grad;
#[cfg(feature = "cuda")]
use crate::ops::shape::{permute, reshape};
use crate::precision::DType;
use ndarray::IxDyn;

fn sample_f32(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 19 + 7) % 31) as f32) / 15.0 - 1.0)
        .collect()
}

fn quantize_bf16(src: &[f32]) -> Vec<f32> {
    src.iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect()
}

fn quantize_f16(src: &[f32]) -> Vec<f32> {
    src.iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect()
}

fn quantize_i8(shape: &[usize], src: &[f32]) -> Vec<f32> {
    let t = make_tensor(shape, src.to_vec(), DType::I8);
    t.data_ref().iter().copied().collect()
}

fn make_tensor(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
    let t = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(shape), data)
            .unwrap()
            .into_dyn(),
    );
    t.cast_inplace(dtype);
    t
}

fn assert_close(lhs: &[f32], rhs: &[f32], tol: f32) {
    assert_eq!(lhs.len(), rhs.len());
    for (idx, (&a, &b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "idx={idx}, lhs={a}, rhs={b}, tol={tol}"
        );
    }
}

#[test]
fn fused_qkv_bf16_matches_quantized_reference() {
    let hidden = 8usize;
    let x = sample_f32(hidden);
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let x_q = quantize_bf16(&x);
    let q_q = quantize_bf16(&q);
    let k_q = quantize_bf16(&k);
    let v_q = quantize_bf16(&v);

    let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
    let input_bf16 = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
    let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
    let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
    let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
    let q_bf16 = make_tensor(&[hidden, hidden], q.clone(), DType::BF16);
    let k_bf16 = make_tensor(&[hidden, hidden], k.clone(), DType::BF16);
    let v_bf16 = make_tensor(&[hidden, hidden], v.clone(), DType::BF16);

    let mut q_ref = vec![0.0f32; hidden];
    let mut k_ref = vec![0.0f32; hidden];
    let mut v_ref = vec![0.0f32; hidden];
    let mut q_out = vec![0.0f32; hidden];
    let mut k_out = vec![0.0f32; hidden];
    let mut v_out = vec![0.0f32; hidden];

    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input_f32, &q_f32, &k_f32, &v_f32, &mut q_ref, &mut k_ref, &mut v_ref,
        );
        fused_qkv_decode_infer_into(
            &input_bf16,
            &q_bf16,
            &k_bf16,
            &v_bf16,
            &mut q_out,
            &mut k_out,
            &mut v_out,
        );
    });

    assert_close(&q_ref, &q_out, 1e-4);
    assert_close(&k_ref, &k_out, 1e-4);
    assert_close(&v_ref, &v_out, 1e-4);
}

#[test]
fn fused_qkv_decode_infer_bf16_matches_quantized_reference() {
    let hidden = 8usize;
    let x = sample_f32(hidden);
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let x_q = quantize_bf16(&x);
    let q_q = quantize_bf16(&q);
    let k_q = quantize_bf16(&k);
    let v_q = quantize_bf16(&v);

    let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
    let input_bf16 = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
    let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
    let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
    let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
    let q_bf16 = make_tensor(&[hidden, hidden], q.clone(), DType::BF16);
    let k_bf16 = make_tensor(&[hidden, hidden], k.clone(), DType::BF16);
    let v_bf16 = make_tensor(&[hidden, hidden], v.clone(), DType::BF16);

    let (q_ref, k_ref, v_ref) =
        no_grad(|| fused_qkv_decode_infer(&input_f32, &q_f32, &k_f32, &v_f32, 1, 1));
    let (q_out, k_out, v_out) =
        no_grad(|| fused_qkv_decode_infer(&input_bf16, &q_bf16, &k_bf16, &v_bf16, 1, 1));

    assert_close(q_ref.as_slice().unwrap(), q_out.as_slice().unwrap(), 1e-4);
    assert_close(k_ref.as_slice().unwrap(), k_out.as_slice().unwrap(), 1e-4);
    assert_close(v_ref.as_slice().unwrap(), v_out.as_slice().unwrap(), 1e-4);
}

#[test]
fn fused_qkv_decode_infer_batch_bf16_matches_quantized_reference() {
    let batch = 2usize;
    let hidden = 8usize;
    let x = sample_f32(batch * hidden)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v + (i / hidden) as f32 * 0.1)
        .collect::<Vec<_>>();
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let x_q = quantize_bf16(&x);
    let q_q = quantize_bf16(&q);
    let k_q = quantize_bf16(&k);
    let v_q = quantize_bf16(&v);

    let input_f32 = make_tensor(&[batch, 1, hidden], x_q.clone(), DType::F32);
    let input_bf16 = make_tensor(&[batch, 1, hidden], x.clone(), DType::BF16);
    let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
    let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
    let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
    let q_bf16 = make_tensor(&[hidden, hidden], q.clone(), DType::BF16);
    let k_bf16 = make_tensor(&[hidden, hidden], k.clone(), DType::BF16);
    let v_bf16 = make_tensor(&[hidden, hidden], v.clone(), DType::BF16);

    let (q_ref, k_ref, v_ref) =
        no_grad(|| fused_qkv_decode_infer(&input_f32, &q_f32, &k_f32, &v_f32, 1, 1));
    let (q_out, k_out, v_out) =
        no_grad(|| fused_qkv_decode_infer(&input_bf16, &q_bf16, &k_bf16, &v_bf16, 1, 1));

    assert_close(q_ref.as_slice().unwrap(), q_out.as_slice().unwrap(), 1e-4);
    assert_close(k_ref.as_slice().unwrap(), k_out.as_slice().unwrap(), 1e-4);
    assert_close(v_ref.as_slice().unwrap(), v_out.as_slice().unwrap(), 1e-4);
}

#[test]
fn fused_qkv_decode_infer_into_accepts_batch_input() {
    let batch = 2usize;
    let hidden = 8usize;
    let x = sample_f32(batch * hidden)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v + (i / hidden) as f32 * 0.15)
        .collect::<Vec<_>>();
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let input = make_tensor(&[batch, 1, hidden], x, DType::BF16);
    let q_w = make_tensor(&[hidden, hidden], q, DType::BF16);
    let k_w = make_tensor(&[hidden, hidden], k, DType::BF16);
    let v_w = make_tensor(&[hidden, hidden], v, DType::BF16);

    let (q_ref, k_ref, v_ref) = no_grad(|| fused_qkv_decode_infer(&input, &q_w, &k_w, &v_w, 1, 1));
    let mut q_out = vec![0.0f32; batch * hidden];
    let mut k_out = vec![0.0f32; batch * hidden];
    let mut v_out = vec![0.0f32; batch * hidden];
    no_grad(|| {
        fused_qkv_decode_infer_into(&input, &q_w, &k_w, &v_w, &mut q_out, &mut k_out, &mut v_out)
    });

    assert_close(q_ref.as_slice().unwrap(), &q_out, 1e-4);
    assert_close(k_ref.as_slice().unwrap(), &k_out, 1e-4);
    assert_close(v_ref.as_slice().unwrap(), &v_out, 1e-4);
}

#[test]
fn fused_qkv_decode_infer_batch_accepts_non_contiguous_input() {
    let batch = 2usize;
    let hidden = 8usize;
    let x = sample_f32(batch * hidden)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v + (i / hidden) as f32 * 0.2)
        .collect::<Vec<_>>();
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let input_contig = make_tensor(&[batch, 1, hidden], x.clone(), DType::F32);
    let input_non_contig = Tensor::from_array_no_grad(
        Array3::from_shape_vec((1, batch, hidden), x)
            .expect("input shape")
            .permuted_axes([1, 0, 2])
            .into_dyn(),
    );
    let q_f32 = make_tensor(&[hidden, hidden], q, DType::F32);
    let k_f32 = make_tensor(&[hidden, hidden], k, DType::F32);
    let v_f32 = make_tensor(&[hidden, hidden], v, DType::F32);

    let (q_ref, k_ref, v_ref) =
        no_grad(|| fused_qkv_decode_infer(&input_contig, &q_f32, &k_f32, &v_f32, 1, 1));
    let (q_out, k_out, v_out) =
        no_grad(|| fused_qkv_decode_infer(&input_non_contig, &q_f32, &k_f32, &v_f32, 1, 1));

    assert_close(q_ref.as_slice().unwrap(), q_out.as_slice().unwrap(), 1e-5);
    assert_close(k_ref.as_slice().unwrap(), k_out.as_slice().unwrap(), 1e-5);
    assert_close(v_ref.as_slice().unwrap(), v_out.as_slice().unwrap(), 1e-5);
}

#[test]
fn fused_qkv_f16_matches_quantized_reference() {
    let hidden = 8usize;
    let x = sample_f32(hidden);
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let x_q = quantize_f16(&x);
    let q_q = quantize_f16(&q);
    let k_q = quantize_f16(&k);
    let v_q = quantize_f16(&v);

    let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
    let input_f16 = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
    let q_f32 = make_tensor(&[hidden, hidden], q_q.clone(), DType::F32);
    let k_f32 = make_tensor(&[hidden, hidden], k_q.clone(), DType::F32);
    let v_f32 = make_tensor(&[hidden, hidden], v_q.clone(), DType::F32);
    let q_f16 = make_tensor(&[hidden, hidden], q.clone(), DType::F16);
    let k_f16 = make_tensor(&[hidden, hidden], k.clone(), DType::F16);
    let v_f16 = make_tensor(&[hidden, hidden], v.clone(), DType::F16);

    let mut q_ref = vec![0.0f32; hidden];
    let mut k_ref = vec![0.0f32; hidden];
    let mut v_ref = vec![0.0f32; hidden];
    let mut q_out = vec![0.0f32; hidden];
    let mut k_out = vec![0.0f32; hidden];
    let mut v_out = vec![0.0f32; hidden];

    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input_f32, &q_f32, &k_f32, &v_f32, &mut q_ref, &mut k_ref, &mut v_ref,
        );
        fused_qkv_decode_infer_into(
            &input_f16, &q_f16, &k_f16, &v_f16, &mut q_out, &mut k_out, &mut v_out,
        );
    });

    assert_close(&q_ref, &q_out, 1e-4);
    assert_close(&k_ref, &k_out, 1e-4);
    assert_close(&v_ref, &v_out, 1e-4);
}

#[test]
fn fused_gateup_bf16_matches_quantized_reference() {
    let hidden = 8usize;
    let inter = 12usize;
    let x = sample_f32(hidden);
    let gate = sample_f32(inter * hidden);
    let up = sample_f32(inter * hidden)
        .into_iter()
        .map(|v| v * 0.5 - 0.2)
        .collect::<Vec<_>>();

    let x_q = quantize_bf16(&x);
    let gate_q = quantize_bf16(&gate);
    let up_q = quantize_bf16(&up);

    let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
    let input_bf16 = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
    let gate_f32 = make_tensor(&[inter, hidden], gate_q.clone(), DType::F32);
    let up_f32 = make_tensor(&[inter, hidden], up_q.clone(), DType::F32);
    let gate_bf16 = make_tensor(&[inter, hidden], gate.clone(), DType::BF16);
    let up_bf16 = make_tensor(&[inter, hidden], up.clone(), DType::BF16);

    let mut ref_out = vec![0.0f32; inter];
    let mut out = vec![0.0f32; inter];

    no_grad(|| {
        fused_gate_up_silu_infer_into(&input_f32, &gate_f32, &up_f32, &mut ref_out);
        fused_gate_up_silu_infer_into(&input_bf16, &gate_bf16, &up_bf16, &mut out);
    });

    assert_close(&ref_out, &out, 1e-3);
}

#[test]
fn fused_gateup_f16_matches_quantized_reference() {
    let hidden = 8usize;
    let inter = 12usize;
    let x = sample_f32(hidden);
    let gate = sample_f32(inter * hidden);
    let up = sample_f32(inter * hidden)
        .into_iter()
        .map(|v| v * 0.5 - 0.2)
        .collect::<Vec<_>>();

    let x_q = quantize_f16(&x);
    let gate_q = quantize_f16(&gate);
    let up_q = quantize_f16(&up);

    let input_f32 = make_tensor(&[1, 1, hidden], x_q.clone(), DType::F32);
    let input_f16 = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
    let gate_f32 = make_tensor(&[inter, hidden], gate_q.clone(), DType::F32);
    let up_f32 = make_tensor(&[inter, hidden], up_q.clone(), DType::F32);
    let gate_f16 = make_tensor(&[inter, hidden], gate.clone(), DType::F16);
    let up_f16 = make_tensor(&[inter, hidden], up.clone(), DType::F16);

    let mut ref_out = vec![0.0f32; inter];
    let mut out = vec![0.0f32; inter];

    no_grad(|| {
        fused_gate_up_silu_infer_into(&input_f32, &gate_f32, &up_f32, &mut ref_out);
        fused_gate_up_silu_infer_into(&input_f16, &gate_f16, &up_f16, &mut out);
    });

    assert_close(&ref_out, &out, 1e-3);
}

#[test]
fn fused_gateup_batch_accepts_non_contiguous_input() {
    let batch = 2usize;
    let hidden = 8usize;
    let inter = 12usize;
    let x = sample_f32(batch * hidden)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v + (i / hidden) as f32 * 0.15)
        .collect::<Vec<_>>();
    let gate = sample_f32(inter * hidden);
    let up = sample_f32(inter * hidden)
        .into_iter()
        .map(|v| v * 0.5 - 0.2)
        .collect::<Vec<_>>();

    let input_contig = make_tensor(&[batch, 1, hidden], x.clone(), DType::F32);
    let input_non_contig = Tensor::from_array_no_grad(
        Array3::from_shape_vec((1, batch, hidden), x)
            .expect("input shape")
            .permuted_axes([1, 0, 2])
            .into_dyn(),
    );
    let gate_f32 = make_tensor(&[inter, hidden], gate, DType::F32);
    let up_f32 = make_tensor(&[inter, hidden], up, DType::F32);

    let ref_out = no_grad(|| fused_gate_up_silu_infer(&input_contig, &gate_f32, &up_f32));
    let out = no_grad(|| fused_gate_up_silu_infer(&input_non_contig, &gate_f32, &up_f32));

    let ref_vals = ref_out.data_ref().iter().copied().collect::<Vec<_>>();
    let out_vals = out.data_ref().iter().copied().collect::<Vec<_>>();
    assert_close(&ref_vals, &out_vals, 1e-5);
}

#[test]
fn fused_qkv_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let x = sample_f32(hidden);
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let q_q = quantize_i8(&[hidden, hidden], &q);
    let k_q = quantize_i8(&[hidden, hidden], &k);
    let v_q = quantize_i8(&[hidden, hidden], &v);

    let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F32);
    let q_ref = make_tensor(&[hidden, hidden], q_q, DType::F32);
    let k_ref = make_tensor(&[hidden, hidden], k_q, DType::F32);
    let v_ref = make_tensor(&[hidden, hidden], v_q, DType::F32);
    let q_i8 = make_tensor(&[hidden, hidden], q, DType::I8);
    let k_i8 = make_tensor(&[hidden, hidden], k, DType::I8);
    let v_i8 = make_tensor(&[hidden, hidden], v, DType::I8);

    let mut q_expected = vec![0.0f32; hidden];
    let mut k_expected = vec![0.0f32; hidden];
    let mut v_expected = vec![0.0f32; hidden];
    let mut q_out = vec![0.0f32; hidden];
    let mut k_out = vec![0.0f32; hidden];
    let mut v_out = vec![0.0f32; hidden];

    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input,
            &q_ref,
            &k_ref,
            &v_ref,
            &mut q_expected,
            &mut k_expected,
            &mut v_expected,
        );
        fused_qkv_decode_infer_into(
            &input, &q_i8, &k_i8, &v_i8, &mut q_out, &mut k_out, &mut v_out,
        );
    });

    assert_close(&q_expected, &q_out, 1e-5);
    assert_close(&k_expected, &k_out, 1e-5);
    assert_close(&v_expected, &v_out, 1e-5);
}

#[test]
fn fused_gateup_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let inter = 12usize;
    let x = sample_f32(hidden);
    let gate = sample_f32(inter * hidden);
    let up = sample_f32(inter * hidden)
        .into_iter()
        .map(|v| v * 0.5 - 0.2)
        .collect::<Vec<_>>();

    let gate_q = quantize_i8(&[inter, hidden], &gate);
    let up_q = quantize_i8(&[inter, hidden], &up);

    let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F32);
    let gate_ref = make_tensor(&[inter, hidden], gate_q, DType::F32);
    let up_ref = make_tensor(&[inter, hidden], up_q, DType::F32);
    let gate_i8 = make_tensor(&[inter, hidden], gate, DType::I8);
    let up_i8 = make_tensor(&[inter, hidden], up, DType::I8);

    let mut ref_out = vec![0.0f32; inter];
    let mut out = vec![0.0f32; inter];

    no_grad(|| {
        fused_gate_up_silu_infer_into(&input, &gate_ref, &up_ref, &mut ref_out);
        fused_gate_up_silu_infer_into(&input, &gate_i8, &up_i8, &mut out);
    });

    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn fused_qkv_bf16_input_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let x = sample_f32(hidden);
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let q_q = quantize_i8(&[hidden, hidden], &q);
    let k_q = quantize_i8(&[hidden, hidden], &k);
    let v_q = quantize_i8(&[hidden, hidden], &v);

    let input = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
    let q_ref = make_tensor(&[hidden, hidden], q_q, DType::F32);
    let k_ref = make_tensor(&[hidden, hidden], k_q, DType::F32);
    let v_ref = make_tensor(&[hidden, hidden], v_q, DType::F32);
    let q_i8 = make_tensor(&[hidden, hidden], q, DType::I8);
    let k_i8 = make_tensor(&[hidden, hidden], k, DType::I8);
    let v_i8 = make_tensor(&[hidden, hidden], v, DType::I8);

    let mut q_expected = vec![0.0f32; hidden];
    let mut k_expected = vec![0.0f32; hidden];
    let mut v_expected = vec![0.0f32; hidden];
    let mut q_out = vec![0.0f32; hidden];
    let mut k_out = vec![0.0f32; hidden];
    let mut v_out = vec![0.0f32; hidden];

    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input,
            &q_ref,
            &k_ref,
            &v_ref,
            &mut q_expected,
            &mut k_expected,
            &mut v_expected,
        );
        fused_qkv_decode_infer_into(
            &input, &q_i8, &k_i8, &v_i8, &mut q_out, &mut k_out, &mut v_out,
        );
    });

    assert_close(&q_expected, &q_out, 1e-5);
    assert_close(&k_expected, &k_out, 1e-5);
    assert_close(&v_expected, &v_out, 1e-5);
}

#[test]
fn fused_qkv_f16_input_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let x = sample_f32(hidden);
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let q_q = quantize_i8(&[hidden, hidden], &q);
    let k_q = quantize_i8(&[hidden, hidden], &k);
    let v_q = quantize_i8(&[hidden, hidden], &v);

    let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
    let q_ref = make_tensor(&[hidden, hidden], q_q, DType::F32);
    let k_ref = make_tensor(&[hidden, hidden], k_q, DType::F32);
    let v_ref = make_tensor(&[hidden, hidden], v_q, DType::F32);
    let q_i8 = make_tensor(&[hidden, hidden], q, DType::I8);
    let k_i8 = make_tensor(&[hidden, hidden], k, DType::I8);
    let v_i8 = make_tensor(&[hidden, hidden], v, DType::I8);

    let mut q_expected = vec![0.0f32; hidden];
    let mut k_expected = vec![0.0f32; hidden];
    let mut v_expected = vec![0.0f32; hidden];
    let mut q_out = vec![0.0f32; hidden];
    let mut k_out = vec![0.0f32; hidden];
    let mut v_out = vec![0.0f32; hidden];

    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input,
            &q_ref,
            &k_ref,
            &v_ref,
            &mut q_expected,
            &mut k_expected,
            &mut v_expected,
        );
        fused_qkv_decode_infer_into(
            &input, &q_i8, &k_i8, &v_i8, &mut q_out, &mut k_out, &mut v_out,
        );
    });

    assert_close(&q_expected, &q_out, 1e-5);
    assert_close(&k_expected, &k_out, 1e-5);
    assert_close(&v_expected, &v_out, 1e-5);
}

#[test]
fn fused_gateup_bf16_input_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let inter = 12usize;
    let x = sample_f32(hidden);
    let gate = sample_f32(inter * hidden);
    let up = sample_f32(inter * hidden)
        .into_iter()
        .map(|v| v * 0.5 - 0.2)
        .collect::<Vec<_>>();

    let gate_q = quantize_i8(&[inter, hidden], &gate);
    let up_q = quantize_i8(&[inter, hidden], &up);

    let input = make_tensor(&[1, 1, hidden], x.clone(), DType::BF16);
    let gate_ref = make_tensor(&[inter, hidden], gate_q, DType::F32);
    let up_ref = make_tensor(&[inter, hidden], up_q, DType::F32);
    let gate_i8 = make_tensor(&[inter, hidden], gate, DType::I8);
    let up_i8 = make_tensor(&[inter, hidden], up, DType::I8);

    let mut ref_out = vec![0.0f32; inter];
    let mut out = vec![0.0f32; inter];

    no_grad(|| {
        fused_gate_up_silu_infer_into(&input, &gate_ref, &up_ref, &mut ref_out);
        fused_gate_up_silu_infer_into(&input, &gate_i8, &up_i8, &mut out);
    });

    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn fused_gateup_f16_input_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let inter = 12usize;
    let x = sample_f32(hidden);
    let gate = sample_f32(inter * hidden);
    let up = sample_f32(inter * hidden)
        .into_iter()
        .map(|v| v * 0.5 - 0.2)
        .collect::<Vec<_>>();

    let gate_q = quantize_i8(&[inter, hidden], &gate);
    let up_q = quantize_i8(&[inter, hidden], &up);

    let input = make_tensor(&[1, 1, hidden], x.clone(), DType::F16);
    let gate_ref = make_tensor(&[inter, hidden], gate_q, DType::F32);
    let up_ref = make_tensor(&[inter, hidden], up_q, DType::F32);
    let gate_i8 = make_tensor(&[inter, hidden], gate, DType::I8);
    let up_i8 = make_tensor(&[inter, hidden], up, DType::I8);

    let mut ref_out = vec![0.0f32; inter];
    let mut out = vec![0.0f32; inter];

    no_grad(|| {
        fused_gate_up_silu_infer_into(&input, &gate_ref, &up_ref, &mut ref_out);
        fused_gate_up_silu_infer_into(&input, &gate_i8, &up_i8, &mut out);
    });

    assert_close(&ref_out, &out, 1e-5);
}

#[test]
fn fused_qkv_i8_input_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let x = sample_f32(hidden)
        .into_iter()
        .map(|v| v * 0.8 - 0.1)
        .collect::<Vec<_>>();
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let x_q = quantize_i8(&[1, 1, hidden], &x);
    let q_q = quantize_i8(&[hidden, hidden], &q);
    let k_q = quantize_i8(&[hidden, hidden], &k);
    let v_q = quantize_i8(&[hidden, hidden], &v);

    let input_ref = make_tensor(&[1, 1, hidden], x_q, DType::F32);
    let input = make_tensor(&[1, 1, hidden], x, DType::I8);
    let q_ref = make_tensor(&[hidden, hidden], q_q, DType::F32);
    let k_ref = make_tensor(&[hidden, hidden], k_q, DType::F32);
    let v_ref = make_tensor(&[hidden, hidden], v_q, DType::F32);
    let q_i8 = make_tensor(&[hidden, hidden], q, DType::I8);
    let k_i8 = make_tensor(&[hidden, hidden], k, DType::I8);
    let v_i8 = make_tensor(&[hidden, hidden], v, DType::I8);

    let mut q_expected = vec![0.0f32; hidden];
    let mut k_expected = vec![0.0f32; hidden];
    let mut v_expected = vec![0.0f32; hidden];
    let mut q_out = vec![0.0f32; hidden];
    let mut k_out = vec![0.0f32; hidden];
    let mut v_out = vec![0.0f32; hidden];

    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input_ref,
            &q_ref,
            &k_ref,
            &v_ref,
            &mut q_expected,
            &mut k_expected,
            &mut v_expected,
        );
        fused_qkv_decode_infer_into(
            &input, &q_i8, &k_i8, &v_i8, &mut q_out, &mut k_out, &mut v_out,
        );
    });

    assert_close(&q_expected, &q_out, 1e-5);
    assert_close(&k_expected, &k_out, 1e-5);
    assert_close(&v_expected, &v_out, 1e-5);
}

#[test]
fn fused_gateup_i8_input_i8_matches_quantized_reference() {
    let hidden = 8usize;
    let inter = 12usize;
    let x = sample_f32(hidden)
        .into_iter()
        .map(|v| v * 0.75 - 0.05)
        .collect::<Vec<_>>();
    let gate = sample_f32(inter * hidden);
    let up = sample_f32(inter * hidden)
        .into_iter()
        .map(|v| v * 0.5 - 0.2)
        .collect::<Vec<_>>();

    let x_q = quantize_i8(&[1, 1, hidden], &x);
    let gate_q = quantize_i8(&[inter, hidden], &gate);
    let up_q = quantize_i8(&[inter, hidden], &up);

    let input_ref = make_tensor(&[1, 1, hidden], x_q, DType::F32);
    let input = make_tensor(&[1, 1, hidden], x, DType::I8);
    let gate_ref = make_tensor(&[inter, hidden], gate_q, DType::F32);
    let up_ref = make_tensor(&[inter, hidden], up_q, DType::F32);
    let gate_i8 = make_tensor(&[inter, hidden], gate, DType::I8);
    let up_i8 = make_tensor(&[inter, hidden], up, DType::I8);

    let mut ref_out = vec![0.0f32; inter];
    let mut out = vec![0.0f32; inter];

    no_grad(|| {
        fused_gate_up_silu_infer_into(&input_ref, &gate_ref, &up_ref, &mut ref_out);
        fused_gate_up_silu_infer_into(&input, &gate_i8, &up_i8, &mut out);
    });

    assert_close(&ref_out, &out, 1e-5);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_gateup_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let inter = 12usize;
    let input = make_tensor(
        &[2, 1, hidden],
        sample_f32(2 * hidden)
            .into_iter()
            .enumerate()
            .map(|(i, v)| v + (i / hidden) as f32 * 0.1)
            .collect(),
        DType::BF16,
    )
    .to_cuda();
    let gate = make_tensor(&[inter, hidden], sample_f32(inter * hidden), DType::BF16).to_cuda();
    let up = make_tensor(
        &[inter, hidden],
        sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect(),
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(gate.0.borrow().cuda_f32_data.is_none());
    assert!(up.0.borrow().cuda_f32_data.is_none());

    crate::autograd::set_strict_device_execution(true);
    let cuda_out = no_grad(|| fused_gate_up_silu_infer(&input, &gate, &up));
    crate::autograd::set_strict_device_execution(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(gate.0.borrow().cuda_f32_data.is_none());
    assert!(up.0.borrow().cuda_f32_data.is_none());

    let cpu_out =
        no_grad(|| fused_gate_up_silu_infer(&input.to_cpu(), &gate.to_cpu(), &up.to_cpu()));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA fused GateUp tensor output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA fused GateUp tensor output should not keep a resident f32 compute buffer for bf16 output"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA fused GateUp tensor output should keep resident bf16 storage"
        );
    }
    assert_eq!(cpu_out.dtype(), DType::BF16);
    assert_eq!(cuda_out.shape_vec(), cpu_out.shape_vec());
    let got = cuda_out.data_ref().iter().copied().collect::<Vec<_>>();
    let expect = cpu_out.data_ref().iter().copied().collect::<Vec<_>>();
    assert_close(&got, &expect, 2e-2);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_gateup_accepts_i8_inputs_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let inter = 12usize;
    let input = make_tensor(
        &[1, 1, hidden],
        sample_f32(hidden)
            .into_iter()
            .map(|v| v * 0.75 - 0.05)
            .collect(),
        DType::I8,
    )
    .to_cuda();
    let gate = make_tensor(&[inter, hidden], sample_f32(inter * hidden), DType::I8).to_cuda();
    let up = make_tensor(
        &[inter, hidden],
        sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect(),
        DType::I8,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(gate.0.borrow().cuda_f32_data.is_none());
    assert!(up.0.borrow().cuda_f32_data.is_none());

    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| fused_gate_up_silu_infer(&input, &gate, &up));
    crate::autograd::set_strict_device_execution(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(gate.0.borrow().cuda_f32_data.is_none());
    assert!(up.0.borrow().cuda_f32_data.is_none());

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::F32);
    let inner = out.0.borrow();
    assert!(
        !inner.has_f32_data,
        "I8 CUDA fused GateUp output should not eagerly materialize host f32 data"
    );
    assert!(
        inner.cuda_f32_data.is_some(),
        "I8 CUDA fused GateUp output should stay resident as f32 compute output"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_gateup_mixed_float_dtypes_promote_to_f32_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let inter = 12usize;
    let input = make_tensor(&[1, 1, hidden], sample_f32(hidden), DType::BF16).to_cuda();
    let gate = make_tensor(&[inter, hidden], sample_f32(inter * hidden), DType::F32).to_cuda();
    let up = make_tensor(
        &[inter, hidden],
        sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect(),
        DType::BF16,
    )
    .to_cuda();

    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| fused_gate_up_silu_infer(&input, &gate, &up));
    crate::autograd::set_strict_device_execution(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::F32);
    assert!(out.cloned_cuda_f32_buffer().is_some());
}

#[test]
fn fused_gateup_i8_or_mixed_dtype_outputs_f32_on_cpu() {
    let hidden = 8usize;
    let inter = 12usize;
    let input = make_tensor(&[1, 1, hidden], sample_f32(hidden), DType::F16);
    let gate = make_tensor(&[inter, hidden], sample_f32(inter * hidden), DType::I8);
    let up = make_tensor(
        &[inter, hidden],
        sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect(),
        DType::I8,
    );

    let out = no_grad(|| fused_gate_up_silu_infer(&input, &gate, &up));

    assert_eq!(out.dtype(), DType::F32);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let batch = 2usize;
    let hidden = 8usize;
    let x = sample_f32(batch * hidden)
        .into_iter()
        .enumerate()
        .map(|(i, v)| v + (i / hidden) as f32 * 0.15)
        .collect::<Vec<_>>();
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let input = make_tensor(&[batch, 1, hidden], x.clone(), DType::BF16).to_cuda();
    let q_w = make_tensor(&[hidden, hidden], q.clone(), DType::BF16).to_cuda();
    let k_w = make_tensor(&[hidden, hidden], k.clone(), DType::BF16).to_cuda();
    let v_w = make_tensor(&[hidden, hidden], v.clone(), DType::BF16).to_cuda();

    crate::autograd::set_strict_device_execution(true);
    let (q_cuda, k_cuda, v_cuda) =
        no_grad(|| fused_qkv_decode_infer(&input, &q_w, &k_w, &v_w, 1, 1));
    crate::autograd::set_strict_device_execution(false);

    let (q_cpu, k_cpu, v_cpu) = no_grad(|| {
        fused_qkv_decode_infer(
            &input.to_cpu(),
            &q_w.to_cpu(),
            &k_w.to_cpu(),
            &v_w.to_cpu(),
            1,
            1,
        )
    });
    assert_close(q_cuda.as_slice().unwrap(), q_cpu.as_slice().unwrap(), 2e-2);
    assert_close(k_cuda.as_slice().unwrap(), k_cpu.as_slice().unwrap(), 2e-2);
    assert_close(v_cuda.as_slice().unwrap(), v_cpu.as_slice().unwrap(), 2e-2);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_accepts_i8_weights_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let x = sample_f32(hidden);
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let input = make_tensor(&[1, 1, hidden], x, DType::I8).to_cuda();
    let q_w = make_tensor(&[hidden, hidden], q, DType::I8).to_cuda();
    let k_w = make_tensor(&[hidden, hidden], k, DType::I8).to_cuda();
    let v_w = make_tensor(&[hidden, hidden], v, DType::I8).to_cuda();

    crate::autograd::set_strict_device_execution(true);
    let (q_out, k_out, v_out) = no_grad(|| fused_qkv_decode_infer(&input, &q_w, &k_w, &v_w, 1, 1));
    crate::autograd::set_strict_device_execution(false);

    assert_eq!(q_out.shape(), &[1, 1, hidden]);
    assert_eq!(k_out.shape(), &[1, 1, hidden]);
    assert_eq!(v_out.shape(), &[1, 1, hidden]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_decode_tensors_preserve_bf16_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let input = make_tensor(&[2, 1, hidden], sample_f32(2 * hidden), DType::BF16).to_cuda();
    let q_w = make_tensor(&[hidden, hidden], sample_f32(hidden * hidden), DType::BF16).to_cuda();
    let k_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect(),
        DType::BF16,
    )
    .to_cuda();
    let v_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect(),
        DType::BF16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(q_w.0.borrow().cuda_f32_data.is_none());
    assert!(k_w.0.borrow().cuda_f32_data.is_none());
    assert!(v_w.0.borrow().cuda_f32_data.is_none());

    crate::autograd::set_strict_device_execution(true);
    let (q_out, k_out, v_out) =
        no_grad(|| fused_qkv_decode_infer_tensors(&input, &q_w, &k_w, &v_w, 1, 1));
    crate::autograd::set_strict_device_execution(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(q_w.0.borrow().cuda_f32_data.is_none());
    assert!(k_w.0.borrow().cuda_f32_data.is_none());
    assert!(v_w.0.borrow().cuda_f32_data.is_none());

    for out in [&q_out, &k_out, &v_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::BF16);
        let inner = out.0.borrow();
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA fused QKV decode output should not keep a resident f32 compute buffer for bf16 output"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA fused QKV decode output should keep resident bf16 storage"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_decode_tensors_mixed_dtype_promotes_to_f32_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let input = make_tensor(&[1, 1, hidden], sample_f32(hidden), DType::F16).to_cuda();
    let q_w = make_tensor(&[hidden, hidden], sample_f32(hidden * hidden), DType::F16).to_cuda();
    let k_w = make_tensor(&[hidden, hidden], sample_f32(hidden * hidden), DType::BF16).to_cuda();
    let v_w = make_tensor(&[hidden, hidden], sample_f32(hidden * hidden), DType::F16).to_cuda();

    crate::autograd::set_strict_device_execution(true);
    let (q_out, k_out, v_out) =
        no_grad(|| fused_qkv_decode_infer_tensors(&input, &q_w, &k_w, &v_w, 1, 1));
    crate::autograd::set_strict_device_execution(false);

    for out in [&q_out, &k_out, &v_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F32);
        assert!(out.cloned_cuda_f32_buffer().is_some());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_prefill_tensors_preserve_f16_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let batch = 2usize;
    let seq = 3usize;
    let hidden = 8usize;
    let input = make_tensor(
        &[batch, seq, hidden],
        sample_f32(batch * seq * hidden),
        DType::F16,
    )
    .to_cuda();
    let q_w = make_tensor(&[hidden, hidden], sample_f32(hidden * hidden), DType::F16).to_cuda();
    let k_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect(),
        DType::F16,
    )
    .to_cuda();
    let v_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect(),
        DType::F16,
    )
    .to_cuda();
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(q_w.0.borrow().cuda_f32_data.is_none());
    assert!(k_w.0.borrow().cuda_f32_data.is_none());
    assert!(v_w.0.borrow().cuda_f32_data.is_none());

    crate::autograd::set_strict_device_execution(true);
    let (q_out, k_out, v_out) = no_grad(|| {
        fused_qkv_prefill_infer_tensors(&input, &q_w, &k_w, &v_w, 1, 1)
            .expect("CUDA prefill fused QKV should run")
    });
    crate::autograd::set_strict_device_execution(false);
    assert!(input.0.borrow().cuda_f32_data.is_none());
    assert!(q_w.0.borrow().cuda_f32_data.is_none());
    assert!(k_w.0.borrow().cuda_f32_data.is_none());
    assert!(v_w.0.borrow().cuda_f32_data.is_none());

    for out in [&q_out, &k_out, &v_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::F16);
        let inner = out.0.borrow();
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA fused QKV prefill output should not keep a resident f32 compute buffer for f16 output"
        );
        assert!(
            inner.cuda_f16_data.is_some(),
            "CUDA fused QKV prefill output should keep resident f16 storage"
        );
    }
    assert_eq!(q_out.shape_vec(), vec![batch, 1, seq, hidden]);
    assert_eq!(k_out.shape_vec(), vec![batch, 1, seq, hidden]);
    assert_eq!(v_out.shape_vec(), vec![batch, 1, seq, hidden]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_prefill_tensors_gqa_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let batch = 1usize;
    let seq = 3usize;
    let hidden = 8usize;
    let kv_hidden = 4usize;
    let input_data = (0..batch * seq * hidden)
        .map(|i| (i as f32 * 0.0625) - 0.4)
        .collect::<Vec<_>>();
    let q_data = (0..hidden * hidden)
        .map(|i| (i as f32 * 0.03125) - 0.25)
        .collect::<Vec<_>>();
    let k_data = (0..kv_hidden * hidden)
        .map(|i| (i as f32 * 0.0275) - 0.25)
        .collect::<Vec<_>>();
    let v_data = (0..kv_hidden * hidden)
        .map(|i| (i as f32 * -0.021) - 0.25)
        .collect::<Vec<_>>();

    let input = make_tensor(&[batch, seq, hidden], input_data, DType::BF16);
    let q_w = make_tensor(&[hidden, hidden], q_data, DType::BF16);
    let k_w = make_tensor(&[kv_hidden, hidden], k_data, DType::BF16);
    let v_w = make_tensor(&[kv_hidden, hidden], v_data, DType::BF16);

    crate::autograd::set_strict_device_execution(true);
    let (q_cuda, k_cuda, v_cuda) = no_grad(|| {
        fused_qkv_prefill_infer_tensors(
            &input.to_cuda(),
            &q_w.to_cuda(),
            &k_w.to_cuda(),
            &v_w.to_cuda(),
            2,
            1,
        )
        .expect("CUDA prefill fused QKV should run")
    });
    crate::autograd::set_strict_device_execution(false);

    let expected_projection = |weight: &Tensor, out_heads: usize| {
        let x_vals = input.data_ref().iter().copied().collect::<Vec<_>>();
        let w_vals = weight.data_ref().iter().copied().collect::<Vec<_>>();
        let out_dim = weight.shape_vec()[0];
        let head_dim = out_dim / out_heads;
        let mut out = vec![0.0f32; batch * out_heads * seq * head_dim];
        for bb in 0..batch {
            for ss in 0..seq {
                for oo in 0..out_dim {
                    let mut sum = 0.0f32;
                    for kk in 0..hidden {
                        sum += x_vals[(bb * seq + ss) * hidden + kk] * w_vals[oo * hidden + kk];
                    }
                    let head = oo / head_dim;
                    let dim = oo % head_dim;
                    out[((bb * out_heads + head) * seq + ss) * head_dim + dim] = sum;
                }
            }
        }
        out
    };
    let q_cpu = expected_projection(&q_w, 2);
    let k_cpu = expected_projection(&k_w, 1);
    let v_cpu = expected_projection(&v_w, 1);

    assert_eq!(q_cuda.shape_vec(), vec![batch, 2, seq, 4]);
    assert_eq!(k_cuda.shape_vec(), vec![batch, 1, seq, 4]);
    assert_eq!(v_cuda.shape_vec(), vec![batch, 1, seq, 4]);
    assert_close(
        &q_cuda.data_ref().iter().copied().collect::<Vec<_>>(),
        &q_cpu,
        2e-2,
    );
    assert_close(
        &k_cuda.data_ref().iter().copied().collect::<Vec<_>>(),
        &k_cpu,
        2e-2,
    );
    assert_close(
        &v_cuda.data_ref().iter().copied().collect::<Vec<_>>(),
        &v_cpu,
        2e-2,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_gateup_into_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let inter = 12usize;
    let input = make_tensor(
        &[1, 1, hidden],
        sample_f32(hidden)
            .into_iter()
            .map(|v| v * 0.9 - 0.1)
            .collect(),
        DType::BF16,
    )
    .to_cuda();
    let gate = make_tensor(&[inter, hidden], sample_f32(inter * hidden), DType::BF16).to_cuda();
    let up = make_tensor(
        &[inter, hidden],
        sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect(),
        DType::BF16,
    )
    .to_cuda();

    let mut cuda_out = vec![0.0f32; inter];
    crate::autograd::set_strict_device_execution(true);
    no_grad(|| fused_gate_up_silu_infer_into(&input, &gate, &up, &mut cuda_out));
    crate::autograd::set_strict_device_execution(false);

    let mut cpu_out = vec![0.0f32; inter];
    no_grad(|| {
        fused_gate_up_silu_infer_into(&input.to_cpu(), &gate.to_cpu(), &up.to_cpu(), &mut cpu_out)
    });
    assert_close(&cuda_out, &cpu_out, 2e-2);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_into_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let x = sample_f32(hidden)
        .into_iter()
        .map(|v| v * 0.85 - 0.05)
        .collect::<Vec<_>>();
    let q = sample_f32(hidden * hidden);
    let k = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v = sample_f32(hidden * hidden)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let input = make_tensor(&[1, 1, hidden], x, DType::BF16).to_cuda();
    let q_w = make_tensor(&[hidden, hidden], q, DType::BF16).to_cuda();
    let k_w = make_tensor(&[hidden, hidden], k, DType::BF16).to_cuda();
    let v_w = make_tensor(&[hidden, hidden], v, DType::BF16).to_cuda();

    let mut q_cuda = vec![0.0f32; hidden];
    let mut k_cuda = vec![0.0f32; hidden];
    let mut v_cuda = vec![0.0f32; hidden];
    crate::autograd::set_strict_device_execution(true);
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input,
            &q_w,
            &k_w,
            &v_w,
            &mut q_cuda,
            &mut k_cuda,
            &mut v_cuda,
        )
    });
    crate::autograd::set_strict_device_execution(false);

    let mut q_cpu = vec![0.0f32; hidden];
    let mut k_cpu = vec![0.0f32; hidden];
    let mut v_cpu = vec![0.0f32; hidden];
    no_grad(|| {
        fused_qkv_decode_infer_into(
            &input.to_cpu(),
            &q_w.to_cpu(),
            &k_w.to_cpu(),
            &v_w.to_cpu(),
            &mut q_cpu,
            &mut k_cpu,
            &mut v_cpu,
        )
    });
    assert_close(&q_cuda, &q_cpu, 2e-2);
    assert_close(&k_cuda, &k_cpu, 2e-2);
    assert_close(&v_cuda, &v_cpu, 2e-2);
}

#[test]
fn fused_softmax_no_grad_preserves_bf16_dtype() {
    let input_f32 = make_tensor(
        &[1, 1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
        DType::F32,
    );
    let input_bf16 = make_tensor(
        &[1, 1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
        DType::BF16,
    );

    let ref_out = no_grad(|| fused_softmax(&input_f32, 1.0, false));
    let out = no_grad(|| fused_softmax(&input_bf16, 1.0, false));

    assert_eq!(input_bf16.dtype(), DType::BF16);
    assert_eq!(out.dtype(), DType::BF16);

    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::F16(_) => {
            panic!("bf16 fused_softmax output should stay bf16 in no-grad")
        }
        TensorStorageView::F32(_) => {
            panic!("bf16 fused_softmax output should stay bf16 in no-grad")
        }
    });
}

#[test]
fn fused_softmax_bf16_matches_quantized_reference() {
    let input_f32 = make_tensor(
        &[1, 1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
        DType::F32,
    );
    let input_bf16 = make_tensor(
        &[1, 1, 2, 4],
        vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, -3.0],
        DType::BF16,
    );

    let ref_out = no_grad(|| fused_softmax(&input_f32, 0.75, true));
    let out = no_grad(|| fused_softmax(&input_bf16, 0.75, true));

    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::F16(_) => {
            panic!("bf16 fused_softmax output should stay bf16 in no-grad")
        }
        TensorStorageView::F32(_) => {
            panic!("bf16 fused_softmax output should stay bf16 in no-grad")
        }
    });
}

#[test]
fn fused_softmax_with_past_no_grad_preserves_bf16_dtype() {
    let input_f32 = make_tensor(
        &[1, 1, 2, 5],
        vec![1.0, -0.5, 2.0, 0.25, -1.25, 0.5, 1.5, -1.0, 0.75, 2.0],
        DType::F32,
    );
    let input_bf16 = make_tensor(
        &[1, 1, 2, 5],
        vec![1.0, -0.5, 2.0, 0.25, -1.25, 0.5, 1.5, -1.0, 0.75, 2.0],
        DType::BF16,
    );

    let ref_out = no_grad(|| fused_softmax_with_past_infer(&input_f32, 0.75, true, 2));
    let out = no_grad(|| fused_softmax_with_past_infer(&input_bf16, 0.75, true, 2));

    assert_eq!(out.dtype(), DType::BF16);
    let ref_vals = ref_out
        .data_ref()
        .iter()
        .map(|&v| half::bf16::from_f32(v).to_f32())
        .collect::<Vec<_>>();
    out.with_storage_view(|view| match view {
        TensorStorageView::BF16(view) => {
            let vals = view.iter().map(|v| v.to_f32()).collect::<Vec<_>>();
            assert_eq!(vals, ref_vals);
        }
        TensorStorageView::F16(_) => {
            panic!("bf16 fused_softmax_with_past output should stay bf16 in no-grad")
        }
        TensorStorageView::F32(_) => {
            panic!("bf16 fused_softmax_with_past output should stay bf16 in no-grad")
        }
    });
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_softmax_preserves_bf16_dtype_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[1, 2, 3, 4],
        vec![
            1.0, -0.5, 2.0, 0.25, 0.5, 1.5, -1.0, 0.75, -0.25, 0.0, 1.25, -1.5, 2.0, -0.75, 0.5,
            1.0, -1.0, 1.25, 0.25, -0.5, 0.75, 0.5, -0.25, 2.5,
        ],
        DType::BF16,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = no_grad(|| fused_softmax(&input, 0.75, true));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = no_grad(|| fused_softmax(&input.to_cpu(), 0.75, true));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA fused_softmax no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA fused_softmax no-grad output should not keep a resident f32 compute buffer for bf16 output"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA fused_softmax no-grad output should keep resident bf16 storage"
        );
    }
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_softmax_promotes_i8_to_f32_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[1, 1, 3, 4],
        vec![
            1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 1.5, -0.5, 2.0, -1.5, 0.75, 0.0,
        ],
        DType::I8,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = no_grad(|| fused_softmax(&input, 1.0, false));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = no_grad(|| fused_softmax(&input.to_cpu(), 1.0, false));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::F32);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "I8 CUDA fused_softmax output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_some(),
            "I8 CUDA fused_softmax output should stay resident as f32 compute output"
        );
    }
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_softmax_backward_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let shape = [1, 2, 3, 4];
    let input_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&shape),
            vec![
                1.0, -0.5, 2.0, 0.25, 0.5, 1.5, -1.0, 0.75, -0.25, 0.0, 1.25, -1.5, 2.0, -0.75,
                0.5, 1.0, -1.0, 1.25, 0.25, -0.5, 0.75, 0.5, -0.25, 2.5,
            ],
        )
        .unwrap()
        .into_dyn(),
        true,
    );
    let coeff_cpu = Tensor::from_data_with_grad_flag(
        Array::from_shape_vec(
            IxDyn(&shape),
            vec![
                -0.3, 0.2, 0.7, -0.1, 0.4, -0.6, 0.8, 0.1, -0.2, 0.5, -0.7, 0.9, 0.25, -0.35, 0.45,
                -0.55, 0.65, -0.75, 0.85, -0.95, 0.15, 0.05, -0.45, 0.55,
            ],
        )
        .unwrap()
        .into_dyn(),
        false,
    );
    let input_cuda = input_cpu.to_cuda();
    let coeff_cuda = coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = fused_softmax(&input_cuda, 0.75, true);
    assert!(cuda_out.is_cuda());
    let loss_cuda = crate::ops::arithmetic::sum(&(&cuda_out * &coeff_cuda));
    loss_cuda.backward();
    assert!(input_cuda.cloned_cuda_f32_grad().is_some());
    assert!(!input_cuda.has_host_grad());
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = fused_softmax(&input_cpu, 0.75, true);
    let loss_cpu = crate::ops::arithmetic::sum(&(&cpu_out * &coeff_cpu));
    loss_cpu.backward();

    let cuda_grad = input_cuda.grad().expect("cuda fused_softmax grad");
    let cpu_grad = input_cpu.grad().expect("cpu fused_softmax grad");
    for (got, expect) in cuda_grad.iter().zip(cpu_grad.iter()) {
        assert!((got - expect).abs() < 1e-5, "got {got}, expect {expect}");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_softmax_with_past_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(
        &[1, 2, 2, 5],
        vec![
            1.0, -0.5, 2.0, 0.25, -1.25, 0.5, 1.5, -1.0, 0.75, 2.0, -0.25, 0.0, 1.25, -1.5, 0.5,
            2.0, -0.75, 0.5, 1.0, -1.0,
        ],
        DType::BF16,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let cuda_out = no_grad(|| fused_softmax_with_past_infer(&input, 0.75, true, 2));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let cpu_out = no_grad(|| fused_softmax_with_past_infer(&input.to_cpu(), 0.75, true, 2));
    assert!(cuda_out.is_cuda());
    assert_eq!(cuda_out.dtype(), DType::BF16);
    {
        let inner = cuda_out.0.borrow();
        assert!(
            !inner.has_f32_data,
            "CUDA fused_softmax_with_past no-grad output should not eagerly materialize host f32 data"
        );
        assert!(
            inner.cuda_f32_data.is_none(),
            "CUDA fused_softmax_with_past no-grad output should not keep a resident f32 compute buffer for bf16 output"
        );
        assert!(
            inner.cuda_bf16_data.is_some(),
            "CUDA fused_softmax_with_past no-grad output should keep resident bf16 storage"
        );
    }
    for (got, expect) in cuda_out.data_ref().iter().zip(cpu_out.data_ref().iter()) {
        assert!((got - expect).abs() < 2e-2, "got {got}, expect {expect}");
    }
}

#[test]
#[should_panic(expected = "Fused Softmax key dimension must be greater than zero")]
fn fused_softmax_rejects_zero_key_dim() {
    let input = make_tensor(&[1, 2, 3, 0], Vec::new(), DType::F32);
    let _ = no_grad(|| fused_softmax(&input, 1.0, false));
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_fused_softmax_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let input = make_tensor(&[1, 2, 0, 4], Vec::new(), DType::BF16).to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| fused_softmax(&input, 1.0, false));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![1, 2, 0, 4]);
    assert_eq!(out.len(), 0);
    assert!(out.cloned_cuda_f32_buffer().is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_fused_gateup_no_grad_stays_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let inter = 12usize;
    let input = make_tensor(&[0, 1, hidden], Vec::new(), DType::BF16).to_cuda();
    let gate = make_tensor(&[inter, hidden], sample_f32(inter * hidden), DType::BF16).to_cuda();
    let up = make_tensor(
        &[inter, hidden],
        sample_f32(inter * hidden)
            .into_iter()
            .map(|v| v * 0.5 - 0.2)
            .collect(),
        DType::BF16,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let out = no_grad(|| fused_gate_up_silu_infer(&input, &gate, &up));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    assert!(out.is_cuda());
    assert_eq!(out.dtype(), DType::BF16);
    assert_eq!(out.shape_vec(), vec![0, 1, inter]);
    assert_eq!(out.len(), 0);
    assert!(out.cloned_cuda_f32_buffer().is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_fused_qkv_decode_tensors_stay_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let hidden = 8usize;
    let input = make_tensor(&[0, 1, hidden], Vec::new(), DType::BF16).to_cuda();
    let q_w = make_tensor(&[hidden, hidden], sample_f32(hidden * hidden), DType::BF16).to_cuda();
    let k_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect(),
        DType::BF16,
    )
    .to_cuda();
    let v_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect(),
        DType::BF16,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let (q_out, k_out, v_out) =
        no_grad(|| fused_qkv_decode_infer_tensors(&input, &q_w, &k_w, &v_w, 1, 1));
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    for out in [&q_out, &k_out, &v_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.len(), 0);
        assert!(out.cloned_cuda_f32_buffer().is_none());
    }
    assert_eq!(q_out.shape_vec(), vec![0, 1, 1, hidden]);
    assert_eq!(k_out.shape_vec(), vec![0, 1, 1, hidden]);
    assert_eq!(v_out.shape_vec(), vec![0, 1, 1, hidden]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_empty_fused_qkv_prefill_tensors_stay_on_cuda_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    let batch = 2usize;
    let seq = 0usize;
    let hidden = 8usize;
    let input = make_tensor(&[batch, seq, hidden], Vec::new(), DType::BF16).to_cuda();
    let q_w = make_tensor(&[hidden, hidden], sample_f32(hidden * hidden), DType::BF16).to_cuda();
    let k_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * 0.7 - 0.1)
            .collect(),
        DType::BF16,
    )
    .to_cuda();
    let v_w = make_tensor(
        &[hidden, hidden],
        sample_f32(hidden * hidden)
            .into_iter()
            .map(|v| v * -0.4 + 0.05)
            .collect(),
        DType::BF16,
    )
    .to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let (q_out, k_out, v_out) = no_grad(|| {
        fused_qkv_prefill_infer_tensors(&input, &q_w, &k_w, &v_w, 1, 1)
            .expect("empty CUDA prefill QKV should return empty tensors")
    });
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    for out in [&q_out, &k_out, &v_out] {
        assert!(out.is_cuda());
        assert_eq!(out.dtype(), DType::BF16);
        assert_eq!(out.len(), 0);
        assert!(out.cloned_cuda_f32_buffer().is_none());
    }
    assert_eq!(q_out.shape_vec(), vec![batch, 1, seq, hidden]);
    assert_eq!(k_out.shape_vec(), vec![batch, 1, seq, hidden]);
    assert_eq!(v_out.shape_vec(), vec![batch, 1, seq, hidden]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_train_bf16_input_i8_weights_backward_stays_native_lowp() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    fn grad_tensor_typed(shape: &[usize], data: Vec<f32>, dtype: DType) -> Tensor {
        let tensor = Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(shape), data)
                .unwrap()
                .into_dyn(),
            true,
        );
        tensor.cast_inplace(dtype);
        tensor
    }

    let b = 2usize;
    let s = 2usize;
    let k_dim = 4usize;
    let n_head = 2usize;
    let n_kv_head = 1usize;
    let d = 4usize;
    let q_n = n_head * d;
    let k_n = n_kv_head * d;

    let input_data = sample_f32(b * s * k_dim);
    let q_data = sample_f32(q_n * k_dim);
    let k_data = sample_f32(k_n * k_dim)
        .into_iter()
        .map(|v| v * 0.7 - 0.1)
        .collect::<Vec<_>>();
    let v_data = sample_f32(k_n * k_dim)
        .into_iter()
        .map(|v| v * -0.4 + 0.05)
        .collect::<Vec<_>>();

    let input_cpu = grad_tensor_typed(&[b, s, k_dim], input_data.clone(), DType::BF16);
    let q_w_cpu = grad_tensor_typed(&[q_n, k_dim], q_data.clone(), DType::I8);
    let k_w_cpu = grad_tensor_typed(&[k_n, k_dim], k_data.clone(), DType::I8);
    let v_w_cpu = grad_tensor_typed(&[k_n, k_dim], v_data.clone(), DType::I8);
    let q_coeff_cpu = make_tensor(
        &[b, n_head, s, d],
        sample_f32(b * n_head * s * d),
        DType::F32,
    );
    let k_coeff_cpu = make_tensor(
        &[b, n_kv_head, s, d],
        sample_f32(b * n_kv_head * s * d),
        DType::F32,
    );
    let v_coeff_cpu = make_tensor(
        &[b, n_kv_head, s, d],
        sample_f32(b * n_kv_head * s * d),
        DType::F32,
    );

    let input_cuda = input_cpu.to_cuda();
    let q_w_cuda = q_w_cpu.to_cuda();
    let k_w_cuda = k_w_cpu.to_cuda();
    let v_w_cuda = v_w_cpu.to_cuda();
    let q_coeff_cuda = q_coeff_cpu.to_cuda();
    let k_coeff_cuda = k_coeff_cpu.to_cuda();
    let v_coeff_cuda = v_coeff_cpu.to_cuda();

    for tensor in [&input_cuda, &q_w_cuda, &k_w_cuda, &v_w_cuda] {
        assert!(tensor.cloned_cuda_native_lowp_buffer().is_some());
        assert!(tensor.0.borrow().cuda_f32_data.is_none());
    }

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let (q_cuda, k_cuda, v_cuda) = fused_qkv_train_tensors(
        &input_cuda,
        &q_w_cuda,
        &k_w_cuda,
        &v_w_cuda,
        None,
        None,
        None,
        n_head,
        n_kv_head,
    )
    .expect("CUDA fused QKV train BF16xI8 should be available");
    crate::ops::arithmetic::sum(&(&q_cuda * &q_coeff_cuda)).backward();
    crate::ops::arithmetic::sum(&(&k_cuda * &k_coeff_cuda)).backward();
    crate::ops::arithmetic::sum(&(&v_cuda * &v_coeff_cuda)).backward();
    for tensor in [&input_cuda, &q_w_cuda, &k_w_cuda, &v_w_cuda] {
        assert!(tensor.cloned_cuda_f32_grad().is_some());
        assert!(
            tensor.0.borrow().cuda_f32_data.is_none(),
            "BF16xI8 fused QKV train backward should not materialize CUDA f32 parameter data"
        );
    }
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let q_cpu = permute(
        &reshape(
            &crate::ops::matmul::matmul(&input_cpu, &q_w_cpu),
            vec![b as i32, s as i32, n_head as i32, d as i32],
        ),
        vec![0, 2, 1, 3],
    );
    let k_cpu = permute(
        &reshape(
            &crate::ops::matmul::matmul(&input_cpu, &k_w_cpu),
            vec![b as i32, s as i32, n_kv_head as i32, d as i32],
        ),
        vec![0, 2, 1, 3],
    );
    let v_cpu = permute(
        &reshape(
            &crate::ops::matmul::matmul(&input_cpu, &v_w_cpu),
            vec![b as i32, s as i32, n_kv_head as i32, d as i32],
        ),
        vec![0, 2, 1, 3],
    );
    crate::ops::arithmetic::sum(&(&q_cpu * &q_coeff_cpu)).backward();
    crate::ops::arithmetic::sum(&(&k_cpu * &k_coeff_cpu)).backward();
    crate::ops::arithmetic::sum(&(&v_cpu * &v_coeff_cpu)).backward();

    for (cuda_tensor, cpu_tensor, label) in [
        (&input_cuda, &input_cpu, "input"),
        (&q_w_cuda, &q_w_cpu, "q weight"),
        (&k_w_cuda, &k_w_cpu, "k weight"),
        (&v_w_cuda, &v_w_cpu, "v weight"),
    ] {
        let cuda_grad = cuda_tensor.grad().expect("cuda grad");
        let cpu_grad = cpu_tensor.grad().expect("cpu grad");
        for (idx, (&got, &expect)) in cuda_grad.iter().zip(cpu_grad.iter()).enumerate() {
            assert!(
                (got - expect).abs() < 3e-3,
                "{label} grad mismatch at {idx}: got {got}, expect {expect}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_fused_qkv_train_with_bias_matches_cpu_reference_in_strict_mode() {
    if !crate::ops::cuda::is_available() {
        return;
    }

    fn grad_tensor(shape: &[usize], data: Vec<f32>) -> Tensor {
        Tensor::from_data_with_grad_flag(
            Array::from_shape_vec(IxDyn(shape), data)
                .unwrap()
                .into_dyn(),
            true,
        )
    }

    let b = 2usize;
    let s = 3usize;
    let k_dim = 4usize;
    let n_head = 2usize;
    let n_kv_head = 1usize;
    let d = 4usize;
    let q_n = n_head * d;
    let k_n = n_kv_head * d;

    let input_cpu = grad_tensor(&[b, s, k_dim], sample_f32(b * s * k_dim));
    let q_w_cpu = grad_tensor(&[q_n, k_dim], sample_f32(q_n * k_dim));
    let k_w_cpu = grad_tensor(&[k_n, k_dim], sample_f32(k_n * k_dim));
    let v_w_cpu = grad_tensor(&[k_n, k_dim], sample_f32(k_n * k_dim));
    let q_b_cpu = grad_tensor(&[q_n], sample_f32(q_n));
    let k_b_cpu = grad_tensor(&[k_n], sample_f32(k_n));
    let v_b_cpu = grad_tensor(&[k_n], sample_f32(k_n));
    let q_coeff_cpu = make_tensor(
        &[b, n_head, s, d],
        sample_f32(b * n_head * s * d),
        DType::F32,
    );
    let k_coeff_cpu = make_tensor(
        &[b, n_kv_head, s, d],
        sample_f32(b * n_kv_head * s * d),
        DType::F32,
    );
    let v_coeff_cpu = make_tensor(
        &[b, n_kv_head, s, d],
        sample_f32(b * n_kv_head * s * d),
        DType::F32,
    );

    let input_cuda = input_cpu.to_cuda();
    let q_w_cuda = q_w_cpu.to_cuda();
    let k_w_cuda = k_w_cpu.to_cuda();
    let v_w_cuda = v_w_cpu.to_cuda();
    let q_b_cuda = q_b_cpu.to_cuda();
    let k_b_cuda = k_b_cpu.to_cuda();
    let v_b_cuda = v_b_cpu.to_cuda();
    let q_coeff_cuda = q_coeff_cpu.to_cuda();
    let k_coeff_cuda = k_coeff_cpu.to_cuda();
    let v_coeff_cuda = v_coeff_cpu.to_cuda();

    crate::ops::cuda::set_enabled(true);
    crate::autograd::set_strict_device_execution(true);
    let (q_cuda, k_cuda, v_cuda) = fused_qkv_train_tensors(
        &input_cuda,
        &q_w_cuda,
        &k_w_cuda,
        &v_w_cuda,
        Some(&q_b_cuda),
        Some(&k_b_cuda),
        Some(&v_b_cuda),
        n_head,
        n_kv_head,
    )
    .expect("CUDA fused QKV train with bias should be available");
    crate::ops::arithmetic::sum(&(&q_cuda * &q_coeff_cuda)).backward();
    crate::ops::arithmetic::sum(&(&k_cuda * &k_coeff_cuda)).backward();
    crate::ops::arithmetic::sum(&(&v_cuda * &v_coeff_cuda)).backward();
    assert!(q_b_cuda.cloned_cuda_f32_grad().is_some());
    assert!(k_b_cuda.cloned_cuda_f32_grad().is_some());
    assert!(v_b_cuda.cloned_cuda_f32_grad().is_some());
    crate::autograd::set_strict_device_execution(false);
    crate::ops::cuda::set_enabled(false);

    let q_cpu = permute(
        &reshape(
            &(crate::ops::matmul::matmul(&input_cpu, &q_w_cpu) + q_b_cpu.clone()),
            vec![b as i32, s as i32, n_head as i32, d as i32],
        ),
        vec![0, 2, 1, 3],
    );
    let k_cpu = permute(
        &reshape(
            &(crate::ops::matmul::matmul(&input_cpu, &k_w_cpu) + k_b_cpu.clone()),
            vec![b as i32, s as i32, n_kv_head as i32, d as i32],
        ),
        vec![0, 2, 1, 3],
    );
    let v_cpu = permute(
        &reshape(
            &(crate::ops::matmul::matmul(&input_cpu, &v_w_cpu) + v_b_cpu.clone()),
            vec![b as i32, s as i32, n_kv_head as i32, d as i32],
        ),
        vec![0, 2, 1, 3],
    );
    crate::ops::arithmetic::sum(&(&q_cpu * &q_coeff_cpu)).backward();
    crate::ops::arithmetic::sum(&(&k_cpu * &k_coeff_cpu)).backward();
    crate::ops::arithmetic::sum(&(&v_cpu * &v_coeff_cpu)).backward();

    let pairs = [
        (&input_cuda, &input_cpu, "input"),
        (&q_w_cuda, &q_w_cpu, "q weight"),
        (&k_w_cuda, &k_w_cpu, "k weight"),
        (&v_w_cuda, &v_w_cpu, "v weight"),
        (&q_b_cuda, &q_b_cpu, "q bias"),
        (&k_b_cuda, &k_b_cpu, "k bias"),
        (&v_b_cuda, &v_b_cpu, "v bias"),
    ];
    for (cuda_tensor, cpu_tensor, label) in pairs {
        let cuda_grad = cuda_tensor.grad().expect("cuda grad");
        let cpu_grad = cpu_tensor.grad().expect("cpu grad");
        for (idx, (&got, &expect)) in cuda_grad.iter().zip(cpu_grad.iter()).enumerate() {
            assert!(
                (got - expect).abs() < 2e-4,
                "{label} grad mismatch at {idx}: got {got}, expect {expect}"
            );
        }
    }
}
