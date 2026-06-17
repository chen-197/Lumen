use super::{Args, Suite, scalar_value};
use lumen::autograd::{Device, Tensor, no_grad, set_strict_device_execution_scoped};
use lumen::init::{ParameterInitMode, with_parameter_init_mode};
use lumen::layers::activation::{Gelu, ReLU, SiLU};
use lumen::layers::{
    Conv2D, Dropout, Embedding, GRU, LSTM, Linear, MaxPool2D, RMSNorm, RNN, SelfAttention,
};
use lumen::loader::{ModelLoader, WeightLoadOptions};
use lumen::loss::{CrossEntropyLoss, MSELoss};
use lumen::models::{LlamaConfig, LlamaModel};
use lumen::module::Module;
use lumen::ops::matmul::batch_matmul;
use lumen::ops::shape::{cat, slice_last_dim};
use lumen::optim::{Adam, Optimizer, SGD};
use lumen::precision::{
    DType, ParameterQuantization, PrecisionConfig, with_precision_config,
    with_runtime_component_dtypes,
};
use lumen::tokenizer::LlamaTokenizer;
use ndarray::{Array, Ix3, IxDyn, s};
use std::path::Path;
use std::time::{Duration, Instant};

fn real_llama_config(max_seq_len: usize) -> LlamaConfig {
    LlamaConfig {
        vocab_size: 32000,
        hidden_size: 2048,
        intermediate_size: 5632,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        rms_norm_eps: 1e-5,
        max_seq_len,
        rope_theta: 10000.0,
    }
}

fn build_first_turn_prompt(system: &str, user: &str) -> String {
    format!(
        "<|system|>\n{}\n</s>\n<|user|>\n{}\n</s>\n<|assistant|>\n",
        system, user
    )
}

fn token_ids_tensor(ids: &[usize], device: Device) -> Tensor {
    Tensor::from_array_no_grad(
        Array::from_shape_vec((1, ids.len()), ids.iter().map(|&id| id as f32).collect())
            .expect("token id tensor shape mismatch")
            .into_dyn(),
    )
    .to_device(device)
}

fn generated_stop_ids(tokenizer: &LlamaTokenizer) -> Vec<usize> {
    let mut stop_ids = Vec::new();
    for token in ["</s>", "<|system|>", "<|user|>", "<|assistant|>"] {
        if let Some(id) = tokenizer.token_to_id(token) {
            stop_ids.push(id);
        }
    }
    if let Some(id) = tokenizer.eos_id() {
        stop_ids.push(id);
    }
    if let Some(id) = tokenizer.eot_id() {
        stop_ids.push(id);
    }
    stop_ids.sort_unstable();
    stop_ids.dedup();
    stop_ids
}

fn last_step_logits_vec(logits: &Tensor) -> Vec<f32> {
    let logits_ref = logits.data_ref();
    let view3 = logits_ref
        .view()
        .into_dimensionality::<Ix3>()
        .expect("logits must be [B, S, V]");
    let last_t = view3.shape()[1] - 1;
    view3.slice(s![0, last_t, ..]).iter().copied().collect()
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (idx, &value) in values.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best_idx = idx;
        }
    }
    best_idx
}

struct TrainingPathStats {
    losses: Vec<f32>,
    elapsed: Duration,
}

impl TrainingPathStats {
    fn steps(&self) -> usize {
        self.losses.len()
    }

    fn us_per_step(&self) -> f64 {
        let steps = self.steps().max(1) as f64;
        self.elapsed.as_secs_f64() * 1e6 / steps
    }

    fn first(&self) -> f32 {
        self.losses.first().copied().unwrap_or_default()
    }

    fn last(&self) -> f32 {
        self.losses.last().copied().unwrap_or_default()
    }

    fn best(&self) -> f32 {
        self.losses
            .iter()
            .copied()
            .fold(f32::INFINITY, |acc, loss| acc.min(loss))
    }
}

fn simple_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(&[4]), vec![-1.0, 0.0, 1.0, 2.0])
            .unwrap()
            .into_dyn(),
    )
    .to_device(device);
    let y = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(&[4]), vec![-1.0, 1.0, 3.0, 5.0])
            .unwrap()
            .into_dyn(),
    )
    .to_device(device);
    let make_param = |value: f32| {
        let data = Array::from_shape_vec(IxDyn(&[1]), vec![value])
            .unwrap()
            .into_dyn();
        if dtype == DType::I8 {
            Tensor::parameter_with_quantization(data, ParameterQuantization::Int8.with_scale(0.02))
        } else {
            Tensor::parameter_with_dtype(data, dtype)
        }
        .to_device(device)
    };
    let w = make_param(0.0);
    let b = make_param(0.0);
    let params = vec![w.clone(), b.clone()];
    let mut opt = SGD::new_with_dtype(params.clone(), 0.08, DType::F32).with_momentum(0.5);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before path training failed");
    }
    let start = Instant::now();
    for _ in 0..24 {
        let pred = &(&x * &w) + &b;
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn tensor_no_grad_with_dtype(
    shape: &[usize],
    values: Vec<f32>,
    dtype: DType,
    device: Device,
) -> Tensor {
    let tensor = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(shape), values)
            .expect("path tensor shape mismatch")
            .into_dyn(),
    );
    if dtype == DType::F32 {
        tensor
    } else {
        let lowp = tensor;
        lowp.cast_inplace(dtype);
        lowp
    }
    .to_device(device)
}

fn parameter_with_dtype(
    shape: &[usize],
    values: Vec<f32>,
    dtype: DType,
    device: Device,
    i8_scale: f32,
) -> Tensor {
    let data = Array::from_shape_vec(IxDyn(shape), values)
        .expect("path parameter shape mismatch")
        .into_dyn();
    if dtype == DType::I8 {
        Tensor::parameter_with_quantization(data, ParameterQuantization::Int8.with_scale(i8_scale))
    } else {
        Tensor::parameter_with_dtype(data, dtype)
    }
    .to_device(device)
}

fn init_linear(linear: &Linear, weight: Vec<f32>, bias: Vec<f32>, dtype: DType) {
    linear.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[linear.out_features, linear.in_features]), weight)
            .expect("path linear weight shape mismatch"),
        dtype,
    );
    linear
        .bias
        .as_ref()
        .expect("path linear bias")
        .set_array_f32_with_dtype(
            Array::from_shape_vec(IxDyn(&[linear.out_features]), bias)
                .expect("path linear bias shape mismatch"),
            dtype,
        );
}

fn init_linear_weight(linear: &Linear, weight: Vec<f32>, dtype: DType, i8_scale: f32) {
    let weight_array =
        Array::from_shape_vec(IxDyn(&[linear.out_features, linear.in_features]), weight)
            .expect("path linear weight shape mismatch");
    if dtype == DType::I8 {
        linear.weight.set_array_f32_with_quantization(
            weight_array,
            ParameterQuantization::Int8.with_scale(i8_scale),
        );
    } else {
        linear.weight.set_array_f32_with_dtype(weight_array, dtype);
    }
}

fn patterned_values(len: usize, scale: f32, phase: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let value = (((idx + phase) % 9) as f32 - 4.0) * scale;
            if value == 0.0 { scale * 0.5 } else { value }
        })
        .collect()
}

fn init_parameter_pattern(param: &Tensor, dtype: DType, scale: f32, phase: usize, i8_scale: f32) {
    let shape = param.shape_vec();
    let len = shape.iter().product::<usize>();
    let values = patterned_values(len, scale, phase);
    let array =
        Array::from_shape_vec(IxDyn(&shape), values).expect("path parameter shape mismatch");
    if dtype == DType::I8 {
        param.set_array_f32_with_quantization(
            array,
            ParameterQuantization::Int8.with_scale(i8_scale),
        );
    } else {
        param.set_array_f32_with_dtype(array, dtype);
    }
}

fn init_conv2d(conv: &Conv2D, weight: Vec<f32>, bias: Vec<f32>, dtype: DType) {
    let out_channels = bias.len();
    let weight_shape = conv.weight.shape_vec();
    if dtype == DType::I8 {
        conv.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(IxDyn(&weight_shape), weight)
                .expect("path conv weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.004),
        );
        conv.bias
            .as_ref()
            .expect("path conv bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[out_channels]), bias)
                    .expect("path conv bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.004),
            );
    } else {
        conv.weight.set_array_f32_with_dtype(
            Array::from_shape_vec(IxDyn(&weight_shape), weight)
                .expect("path conv weight shape mismatch"),
            dtype,
        );
        conv.bias
            .as_ref()
            .expect("path conv bias")
            .set_array_f32_with_dtype(
                Array::from_shape_vec(IxDyn(&[out_channels]), bias)
                    .expect("path conv bias shape mismatch"),
                dtype,
            );
    }
}

fn one_hot_tensor(rows: usize, cols: usize, labels: &[usize], device: Device) -> Tensor {
    assert_eq!(rows, labels.len(), "one-hot label count mismatch");
    let mut data = vec![0.0f32; rows * cols];
    for (row, &label) in labels.iter().enumerate() {
        assert!(label < cols, "one-hot label out of range");
        data[row * cols + label] = 1.0;
    }
    Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(&[rows, cols]), data)
            .expect("one-hot target shape mismatch")
            .into_dyn(),
    )
    .to_device(device)
}

fn assert_cuda_f32_grads_without_host(params: &[Tensor], name: &str) {
    assert!(
        params.iter().all(Tensor::dev_has_cuda_f32_grad),
        "{name}.cuda expected CUDA f32 gradients after backward"
    );
    assert!(
        params.iter().all(|param| !param.dev_has_host_grad()),
        "{name}.cuda should not materialize host gradients during strict CUDA training"
    );
}

fn assert_cuda_f32_momentum_state(opt: &SGD, param_count: usize, name: &str) {
    assert_eq!(
        opt.dev_velocity_count(),
        param_count,
        "{name}.cuda expected one momentum velocity per parameter"
    );
    assert!(
        opt.dev_all_velocities_are_f32_cuda_resident(),
        "{name}.cuda expected f32 CUDA-resident momentum state"
    );
}

fn assert_cuda_f32_adam_state(opt: &Adam, param_count: usize, name: &str) {
    assert_eq!(
        opt.dev_state_pair_count(),
        param_count,
        "{name}.cuda expected one Adam state pair per parameter"
    );
    assert!(
        opt.dev_all_states_are_f32_cuda_resident(),
        "{name}.cuda expected f32 CUDA-resident Adam state"
    );
}

fn mlp_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[4, 3],
        vec![0.2, 0.5, 1.0, 0.4, 0.1, 0.8, 0.7, 0.3, 0.6, 1.0, 0.2, 0.4],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[4, 2],
        vec![0.32, 0.18, 0.26, 0.22, 0.35, 0.28, 0.42, 0.24],
        dtype,
        device,
    );

    let fc1 = Linear::new_with_dtype(3, 5, dtype);
    init_linear(
        &fc1,
        vec![
            0.18, 0.05, -0.04, 0.10, 0.14, 0.08, -0.06, 0.09, 0.16, 0.12, -0.03, 0.04, 0.07, 0.11,
            0.15,
        ],
        vec![0.08, 0.04, 0.06, 0.05, 0.07],
        dtype,
    );
    let fc2 = Linear::new_with_dtype(5, 2, dtype);
    init_linear(
        &fc2,
        vec![0.12, -0.03, 0.08, 0.04, 0.10, 0.05, 0.09, -0.02, 0.11, 0.06],
        vec![0.02, 0.03],
        dtype,
    );
    if device == Device::Cuda {
        fc1.to_cuda();
        fc2.to_cuda();
    }

    let mut params = fc1.parameters();
    params.extend(fc2.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.03, DType::F32).with_momentum(0.4);
    let relu = ReLU::new();
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before MLP path training failed");
    }
    let start = Instant::now();
    for _ in 0..40 {
        let hidden = relu.forward(fc1.forward(x.clone()));
        let pred = fc2.forward(hidden);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.mlp");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.mlp");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after MLP path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn gelu_mlp_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.25, -0.15, 0.35, 0.05, -0.20, 0.30, 0.10, 0.28, 0.32, 0.04, -0.22, 0.16, -0.12, 0.24,
            0.18, -0.06,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[4, 3],
        vec![
            0.12, 0.04, 0.18, 0.06, 0.16, 0.08, 0.20, -0.02, 0.10, 0.04, 0.14, 0.12,
        ],
        dtype,
        device,
    );

    let fc1 = Linear::new_with_dtype(4, 6, dtype);
    if dtype == DType::I8 {
        fc1.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(IxDyn(&[6, 4]), patterned_values(24, 0.080, 1))
                .expect("GELU path fc1 weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.004),
        );
        fc1.bias
            .as_ref()
            .expect("GELU path fc1 bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[6]), patterned_values(6, 0.030, 2))
                    .expect("GELU path fc1 bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.002),
            );
    } else {
        init_linear(
            &fc1,
            patterned_values(24, 0.080, 1),
            patterned_values(6, 0.030, 2),
            dtype,
        );
    }
    let fc2 = Linear::new_with_dtype(6, 3, dtype);
    if dtype == DType::I8 {
        fc2.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(IxDyn(&[3, 6]), patterned_values(18, 0.060, 4))
                .expect("GELU path fc2 weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.003),
        );
        fc2.bias
            .as_ref()
            .expect("GELU path fc2 bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[3]), vec![0.0, 0.01, -0.01])
                    .expect("GELU path fc2 bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.001),
            );
    } else {
        init_linear(
            &fc2,
            patterned_values(18, 0.060, 4),
            vec![0.0, 0.01, -0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        fc1.to_cuda();
        fc2.to_cuda();
    }

    let mut params = fc1.parameters();
    params.extend(fc2.parameters());
    let lr = if dtype == DType::I8 { 0.45 } else { 0.12 };
    let mut opt = SGD::new_with_dtype(params.clone(), lr, DType::F32).with_momentum(0.25);
    let gelu = Gelu::new();
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before GELU MLP path training failed");
    }
    let start = Instant::now();
    for _ in 0..42 {
        let hidden = gelu.forward(fc1.forward(x.clone()));
        let pred = fc2.forward(hidden);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.gelu_mlp");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.gelu_mlp");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after GELU MLP path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn dropout_mlp_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.30, -0.18, 0.24, 0.06, -0.16, 0.34, 0.08, 0.22, 0.28, 0.02, -0.24, 0.14, -0.08, 0.26,
            0.16, -0.04,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[4, 2],
        vec![0.10, 0.02, 0.04, 0.14, 0.18, -0.02, 0.06, 0.12],
        dtype,
        device,
    );

    let fc1 = Linear::new_with_dtype(4, 6, dtype);
    if dtype == DType::I8 {
        fc1.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(IxDyn(&[6, 4]), patterned_values(24, 0.075, 8))
                .expect("Dropout path fc1 weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.004),
        );
        fc1.bias
            .as_ref()
            .expect("Dropout path fc1 bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[6]), patterned_values(6, 0.025, 9))
                    .expect("Dropout path fc1 bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.002),
            );
    } else {
        init_linear(
            &fc1,
            patterned_values(24, 0.075, 8),
            patterned_values(6, 0.025, 9),
            dtype,
        );
    }
    let fc2 = Linear::new_with_dtype(6, 2, dtype);
    if dtype == DType::I8 {
        fc2.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(IxDyn(&[2, 6]), patterned_values(12, 0.055, 10))
                .expect("Dropout path fc2 weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.003),
        );
        fc2.bias
            .as_ref()
            .expect("Dropout path fc2 bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("Dropout path fc2 bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.001),
            );
    } else {
        init_linear(
            &fc2,
            patterned_values(12, 0.055, 10),
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        fc1.to_cuda();
        fc2.to_cuda();
    }

    let mut params = fc1.parameters();
    params.extend(fc2.parameters());
    let lr = if dtype == DType::I8 { 0.38 } else { 0.10 };
    let mut opt = SGD::new_with_dtype(params.clone(), lr, DType::F32).with_momentum(0.25);
    let relu = ReLU::new();
    let dropout = Dropout::new_with_seed(0.25, 0xD00D);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before Dropout MLP path training failed");
    }
    let start = Instant::now();
    for _ in 0..44 {
        let hidden = relu.forward(fc1.forward(x.clone()));
        let dropped = dropout.forward(hidden);
        let pred = fc2.forward(dropped);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.dropout");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.dropout");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after Dropout MLP path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn batch_matmul_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let lhs = tensor_no_grad_with_dtype(
        &[2, 2, 2, 3],
        vec![
            0.20, -0.10, 0.30, 0.05, 0.25, -0.15, -0.12, 0.18, 0.22, 0.28, -0.06, 0.14, 0.16, 0.24,
            -0.08, -0.20, 0.10, 0.26, 0.32, -0.04, 0.12, 0.06, 0.22, -0.18,
        ],
        dtype,
        device,
    );
    let target = tensor_no_grad_with_dtype(
        &[2, 2, 2, 2],
        vec![
            0.05, 0.14, 0.11, 0.02, -0.04, 0.10, 0.08, 0.12, 0.16, -0.02, 0.04, 0.18, 0.10, 0.06,
            -0.03, 0.13,
        ],
        dtype,
        device,
    );
    let rhs_init = Array::from_shape_vec(
        IxDyn(&[2, 2, 3, 2]),
        vec![
            0.12, -0.04, 0.08, 0.10, -0.06, 0.14, 0.05, 0.16, 0.11, -0.02, 0.09, 0.07, -0.10, 0.12,
            0.04, 0.15, 0.13, -0.05, 0.06, 0.08, -0.03, 0.11, 0.14, 0.02,
        ],
    )
    .expect("BatchMatMul path rhs shape mismatch");
    let rhs = if dtype == DType::I8 {
        Tensor::parameter_with_quantization(rhs_init, ParameterQuantization::Int8.with_scale(0.003))
    } else {
        Tensor::parameter_with_dtype(rhs_init, dtype)
    }
    .to_device(device);

    let params = vec![rhs.clone()];
    let lr = match dtype {
        DType::F32 | DType::F16 => 0.45,
        DType::BF16 => 0.42,
        DType::I8 => 0.55,
    };
    let mut opt = SGD::new_with_dtype(params.clone(), lr, DType::F32).with_momentum(0.2);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before BatchMatMul path training failed");
    }
    let start = Instant::now();
    for _ in 0..42 {
        let pred = batch_matmul(&lhs, &rhs);
        let loss = MSELoss::apply(&pred, &target);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.batch_matmul");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.batch_matmul");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after BatchMatMul path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn gated_mlp_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.18, -0.12, 0.32, 0.05, -0.22, 0.28, 0.10, 0.24, 0.30, 0.06, -0.18, 0.14, -0.10, 0.20,
            0.26, -0.08,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.12, 0.02, 0.18, -0.04, 0.06, 0.16, 0.04, 0.12, 0.20, -0.02, 0.10, 0.08, 0.04, 0.14,
            0.06, 0.18,
        ],
        dtype,
        device,
    );

    let gate = Linear::new_no_bias_with_dtype(4, 6, dtype);
    let up = Linear::new_no_bias_with_dtype(4, 6, dtype);
    let down = Linear::new_no_bias_with_dtype(6, 4, dtype);
    init_linear_weight(&gate, patterned_values(24, 0.400, 0), dtype, 0.020);
    init_linear_weight(&up, patterned_values(24, 0.300, 3), dtype, 0.020);
    init_linear_weight(&down, patterned_values(24, 0.200, 6), dtype, 0.020);
    if device == Device::Cuda {
        gate.to_cuda();
        up.to_cuda();
        down.to_cuda();
    }

    let mut params = gate.parameters();
    params.extend(up.parameters());
    params.extend(down.parameters());
    let lr = if dtype == DType::I8 { 3.0 } else { 0.22 };
    let mut opt = SGD::new_with_dtype(params.clone(), lr, DType::F32).with_momentum(0.25);
    let silu = SiLU::new();
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before gated-MLP path training failed");
    }
    let start = Instant::now();
    for _ in 0..48 {
        let gate_act = silu.forward(gate.forward(x.clone()));
        let up_out = up.forward(x.clone());
        let pred = down.forward(gate_act * up_out);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.gated_mlp");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.gated_mlp");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after gated-MLP path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn residual_mix_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.70, -0.42, 0.54, 0.18, -0.50, 0.64, 0.22, 0.46, 0.58, 0.14, -0.38, 0.34, -0.24, 0.48,
            0.40, -0.18,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.34, -0.18, 0.26, 0.10, -0.08, 0.30, 0.18, 0.22, 0.38, -0.04, 0.24, 0.16, 0.12, 0.28,
            0.06, 0.32,
        ],
        dtype,
        device,
    );

    let trunk = Linear::new_no_bias_with_dtype(4, 4, dtype);
    let skip = Linear::new_no_bias_with_dtype(4, 4, dtype);
    let gate = Linear::new_no_bias_with_dtype(4, 4, dtype);
    init_linear_weight(&trunk, patterned_values(16, 0.120, 11), dtype, 0.005);
    init_linear_weight(&skip, patterned_values(16, 0.080, 13), dtype, 0.004);
    init_linear_weight(&gate, patterned_values(16, 0.140, 17), dtype, 0.005);
    if device == Device::Cuda {
        trunk.to_cuda();
        skip.to_cuda();
        gate.to_cuda();
    }

    let mut params = trunk.parameters();
    params.extend(skip.parameters());
    params.extend(gate.parameters());
    let lr = match dtype {
        DType::F32 | DType::F16 => 1.0,
        DType::BF16 => 1.1,
        DType::I8 => 8.0,
    };
    let mut opt = SGD::new_with_dtype(params.clone(), lr, DType::F32).with_momentum(0.20);
    let silu = SiLU::new();
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync before residual-mix path training failed");
    }
    let start = Instant::now();
    for _ in 0..72 {
        let trunk_out = trunk.forward(x.clone());
        let skip_out = skip.forward(x.clone());
        let gate_act = silu.forward(gate.forward(x.clone()));
        let pred = (trunk_out + skip_out) * gate_act;
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.residual_mix");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.residual_mix");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after residual-mix path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn broadcast_affine_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));
    let runtime_dtype = if dtype == DType::I8 {
        DType::BF16
    } else {
        dtype
    };

    let x = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.42, -0.26, 0.36, 0.18, -0.32, 0.48, 0.22, 0.30, 0.54, 0.12, -0.40, 0.28, -0.18, 0.34,
            0.46, -0.12,
        ],
        runtime_dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.24, -0.08, 0.22, 0.12, -0.04, 0.26, 0.16, 0.18, 0.30, 0.02, 0.20, 0.14, 0.08, 0.22,
            0.10, 0.24,
        ],
        runtime_dtype,
        device,
    );

    let scale = parameter_with_dtype(&[1, 4], vec![0.55, 0.35, 0.45, 0.60], dtype, device, 0.005);
    let bias = parameter_with_dtype(&[1, 4], vec![0.02, -0.04, 0.03, 0.01], dtype, device, 0.001);
    let params = vec![scale.clone(), bias.clone()];
    let lr = match dtype {
        DType::F32 | DType::F16 => 0.35,
        DType::BF16 => 0.38,
        DType::I8 => 1.8,
    };
    let mut opt = SGD::new_with_dtype(params.clone(), lr, DType::F32).with_momentum(0.20);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync before broadcast-affine path training failed");
    }
    let start = Instant::now();
    for _ in 0..64 {
        let pred = &(&x * &scale) + &bias;
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.broadcast_affine");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.broadcast_affine");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync after broadcast-affine path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn classifier_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let vocab_size = 6usize;
    let embed_dim = 4usize;
    let class_count = 3usize;
    let token_count = 6usize;
    let token_ids = Tensor::from_array_no_grad(
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("token id shape mismatch")
            .into_dyn(),
    )
    .to_device(device);
    let targets = one_hot_tensor(token_count, class_count, &[0, 1, 2, 0, 1, 2], device);

    let embedding = Embedding::new_with_dtype(vocab_size, embed_dim, dtype);
    embedding.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(
            IxDyn(&[vocab_size, embed_dim]),
            vec![
                0.10, 0.20, -0.10, 0.05, -0.15, 0.25, 0.05, 0.10, 0.20, -0.05, 0.15, -0.10, -0.05,
                0.10, 0.25, 0.15, 0.12, -0.20, 0.18, 0.08, -0.22, 0.05, 0.10, 0.20,
            ],
        )
        .expect("embedding weight shape mismatch"),
        dtype,
    );
    let head = Linear::new_with_dtype(embed_dim, class_count, dtype);
    init_linear(
        &head,
        vec![
            0.15, -0.05, 0.10, 0.08, -0.10, 0.12, 0.06, -0.04, 0.05, 0.08, -0.12, 0.14,
        ],
        vec![0.01, -0.02, 0.03],
        dtype,
    );
    if device == Device::Cuda {
        embedding.to_cuda();
        head.to_cuda();
    }

    let mut params = embedding.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.12, DType::F32).with_momentum(0.3);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before classifier path training failed");
    }
    let start = Instant::now();
    for _ in 0..36 {
        let embeddings = embedding
            .forward(&token_ids)
            .reshape(vec![token_count as i32, embed_dim as i32]);
        let logits = head.forward(embeddings);
        let loss = CrossEntropyLoss::apply(&logits, &targets);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.classifier");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.classifier");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after classifier path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn shape_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[2, 3],
        vec![0.30, -0.20, 0.45, -0.10, 0.25, 0.15],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[2, 4],
        vec![0.18, 0.12, 0.34, 0.26, 0.08, 0.28, 0.16, 0.38],
        dtype,
        device,
    );

    let lhs = Linear::new_with_dtype(3, 4, dtype);
    init_linear(
        &lhs,
        vec![
            0.10, -0.20, 0.30, 0.05, 0.25, -0.15, -0.10, 0.35, 0.20, 0.15, -0.05, 0.22,
        ],
        vec![0.02, -0.01, 0.03, 0.00],
        dtype,
    );
    let rhs = Linear::new_with_dtype(3, 4, dtype);
    init_linear(
        &rhs,
        vec![
            -0.12, 0.18, 0.24, 0.16, -0.22, 0.08, 0.26, 0.10, -0.18, 0.04, 0.28, -0.06,
        ],
        vec![-0.01, 0.02, -0.02, 0.01],
        dtype,
    );
    if device == Device::Cuda {
        lhs.to_cuda();
        rhs.to_cuda();
    }

    let mut params = lhs.parameters();
    params.extend(rhs.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.06, DType::F32).with_momentum(0.20);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before shape path training failed");
    }
    let start = Instant::now();
    for _ in 0..40 {
        let lhs_view = lhs
            .forward(x.clone())
            .reshape(vec![2, 2, 2])
            .permute(vec![0, 2, 1])
            .reshape(vec![2, 4]);
        let rhs_view = rhs
            .forward(x.clone())
            .reshape(vec![2, 2, 2])
            .permute(vec![0, 2, 1])
            .reshape(vec![2, 4]);
        let joined = cat(&[lhs_view, rhs_view], 1);
        let left = slice_last_dim(&joined, 0, 4);
        let right = slice_last_dim(&joined, 4, 8);
        let pred = &left + &right;
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.shape");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.shape");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after shape path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn shared_parameter_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x_a = tensor_no_grad_with_dtype(
        &[3, 3],
        vec![0.40, -0.10, 0.20, -0.25, 0.35, 0.15, 0.10, 0.30, -0.20],
        dtype,
        device,
    );
    let x_b = tensor_no_grad_with_dtype(
        &[3, 3],
        vec![-0.15, 0.25, 0.30, 0.20, -0.35, 0.10, 0.45, 0.05, -0.25],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[3, 4],
        vec![
            0.28, 0.10, 0.36, 0.18, 0.12, 0.32, 0.16, 0.24, 0.34, 0.14, 0.22, 0.30,
        ],
        dtype,
        device,
    );

    let shared = Linear::new_with_dtype(3, 4, dtype);
    init_linear(
        &shared,
        vec![
            0.08, -0.16, 0.24, 0.12, 0.20, -0.10, -0.18, 0.14, 0.26, 0.22, -0.06, 0.10,
        ],
        vec![0.00, 0.02, -0.01, 0.01],
        dtype,
    );
    if device == Device::Cuda {
        shared.to_cuda();
    }

    let params = shared.parameters();
    let mut opt = SGD::new_with_dtype(params.clone(), 0.08, DType::F32).with_momentum(0.25);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync before shared-param path training failed");
    }
    let start = Instant::now();
    for _ in 0..44 {
        let first_use = shared.forward(x_a.clone());
        let second_use = shared.forward(x_b.clone());
        let pred = &first_use + &second_use;
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.shared_param");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.shared_param");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after shared-param path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn gradient_accumulation_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x_a = tensor_no_grad_with_dtype(
        &[2, 3],
        vec![0.35, -0.15, 0.25, -0.20, 0.30, 0.10],
        dtype,
        device,
    );
    let y_a = tensor_no_grad_with_dtype(&[2, 2], vec![0.22, 0.14, 0.10, 0.26], dtype, device);
    let x_b = tensor_no_grad_with_dtype(
        &[2, 3],
        vec![-0.10, 0.28, 0.18, 0.42, -0.08, 0.12],
        dtype,
        device,
    );
    let y_b = tensor_no_grad_with_dtype(&[2, 2], vec![0.16, 0.30, 0.32, 0.12], dtype, device);

    let head = Linear::new_with_dtype(3, 2, dtype);
    init_linear(
        &head,
        vec![0.12, -0.08, 0.20, -0.16, 0.22, 0.10],
        vec![0.02, -0.01],
        dtype,
    );
    if device == Device::Cuda {
        head.to_cuda();
    }

    let params = head.parameters();
    let mut opt = SGD::new_with_dtype(params.clone(), 0.09, DType::F32).with_momentum(0.20);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before grad-accum path training failed");
    }
    let start = Instant::now();
    for _ in 0..42 {
        let loss_a = MSELoss::apply(&head.forward(x_a.clone()), &y_a);
        let loss_b = MSELoss::apply(&head.forward(x_b.clone()), &y_b);
        losses.push((scalar_value(&loss_a) + scalar_value(&loss_b)) * 0.5);

        loss_a.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.grad_accum.first_backward");
        }
        loss_b.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.grad_accum.second_backward");
        }

        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.grad_accum");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after grad-accum path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn optimizer_batch_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let batch = 4usize;
    let dim = 192usize;
    let shard_count = 8usize;
    let x = tensor_no_grad_with_dtype(
        &[batch, dim],
        patterned_values(batch * dim, 0.025, 3),
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[batch, dim],
        patterned_values(batch * dim, 0.080, 11),
        dtype,
        device,
    );

    let mut shards = Vec::new();
    for shard in 0..shard_count {
        let layer = Linear::new_with_dtype(dim, dim, dtype);
        init_linear(
            &layer,
            patterned_values(dim * dim, 0.0003, shard),
            patterned_values(dim, 0.0002, shard + 5),
            dtype,
        );
        if device == Device::Cuda {
            layer.to_cuda();
        }
        shards.push(layer);
    }

    let mut params = Vec::new();
    for shard in &shards {
        params.extend(shard.parameters());
    }
    let mut opt = SGD::new_with_dtype(params.clone(), 0.08, DType::F32).with_momentum(0.15);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync before optimizer-batch path training failed");
    }
    let start = Instant::now();
    for _ in 0..24 {
        let mut pred = shards[0].forward(x.clone());
        for shard in shards.iter().skip(1) {
            pred = &pred + &shard.forward(x.clone());
        }
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.optimizer_batch");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.optimizer_batch");
            assert!(
                opt.dev_last_cuda_batched_update_count() >= shard_count,
                "path.train.optimizer_batch expected CUDA batched optimizer update for at least {shard_count} params, got {}",
                opt.dev_last_cuda_batched_update_count()
            );
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync after optimizer-batch path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn adam_batch_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let batch = 4usize;
    let dim = 192usize;
    let shard_count = 8usize;
    let x = tensor_no_grad_with_dtype(
        &[batch, dim],
        patterned_values(batch * dim, 0.020, 7),
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[batch, dim],
        patterned_values(batch * dim, 0.070, 17),
        dtype,
        device,
    );

    let mut shards = Vec::new();
    for shard in 0..shard_count {
        let layer = Linear::new_with_dtype(dim, dim, dtype);
        init_linear(
            &layer,
            patterned_values(dim * dim, 0.00025, shard + 13),
            patterned_values(dim, 0.00015, shard + 19),
            dtype,
        );
        if device == Device::Cuda {
            layer.to_cuda();
        }
        shards.push(layer);
    }

    let mut params = Vec::new();
    for shard in &shards {
        params.extend(shard.parameters());
    }
    let mut opt = Adam::new_with_dtype(params.clone(), 0.006, DType::F32);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before Adam-batch path training failed");
    }
    let start = Instant::now();
    for _ in 0..24 {
        let mut pred = shards[0].forward(x.clone());
        for shard in shards.iter().skip(1) {
            pred = &pred + &shard.forward(x.clone());
        }
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.adam_batch");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_adam_state(&opt, params.len(), "path.train.adam_batch");
            assert!(
                opt.dev_last_cuda_batched_update_count() >= shard_count,
                "path.train.adam_batch expected CUDA batched Adam update for at least {shard_count} params, got {}",
                opt.dev_last_cuda_batched_update_count()
            );
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after Adam-batch path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn norm_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[4, 4],
        vec![
            0.25, -0.10, 0.35, 0.70, -0.40, 0.20, 0.55, -0.15, 0.80, 0.10, -0.30, 0.45, -0.20,
            0.65, 0.30, 0.05,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[4, 2],
        vec![0.18, 0.42, 0.24, 0.16, 0.38, 0.28, 0.12, 0.32],
        dtype,
        device,
    );

    let norm = RMSNorm::new_with_dtype(4, 1e-5, dtype);
    norm.weight.set_array_f32_with_dtype(
        Array::from_shape_vec(IxDyn(&[4]), vec![0.95, 1.05, 0.90, 1.10])
            .expect("RMSNorm weight shape mismatch"),
        dtype,
    );
    let head = Linear::new_with_dtype(4, 2, dtype);
    init_linear(
        &head,
        vec![0.08, -0.03, 0.10, 0.05, -0.04, 0.09, 0.02, 0.11],
        vec![0.03, 0.02],
        dtype,
    );
    if device == Device::Cuda {
        norm.to_cuda();
        head.to_cuda();
    }

    let mut params = norm.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.025, DType::F32).with_momentum(0.35);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before norm path training failed");
    }
    let start = Instant::now();
    for _ in 0..40 {
        let pred = head.forward(norm.forward(x.clone()));
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.norm");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.norm");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after norm path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn recurrent_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[3, 3],
        vec![0.20, -0.10, 0.35, -0.30, 0.15, 0.25, 0.10, 0.40, -0.20],
        dtype,
        device,
    );
    let h_prev = tensor_no_grad_with_dtype(
        &[3, 4],
        vec![
            0.05, -0.10, 0.15, 0.20, -0.15, 0.25, 0.10, -0.05, 0.30, 0.05, -0.20, 0.15,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[3, 2],
        vec![0.12, 0.18, 0.08, 0.22, 0.16, 0.10],
        dtype,
        device,
    );

    let rnn = RNN::new_with_dtype(3, 4, dtype);
    for (idx, param) in rnn.parameters().iter().enumerate() {
        init_parameter_pattern(param, dtype, 0.015, idx, 0.001);
    }
    let head = Linear::new_with_dtype(4, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(
                IxDyn(&[2, 4]),
                vec![0.03, -0.01, 0.04, 0.02, -0.02, 0.05, 0.01, 0.03],
            )
            .expect("recurrent path head weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.001),
        );
        head.bias
            .as_ref()
            .expect("recurrent path head bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("recurrent path head bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.001),
            );
    } else {
        init_linear(
            &head,
            vec![0.03, -0.01, 0.04, 0.02, -0.02, 0.05, 0.01, 0.03],
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        rnn.to_cuda();
        head.to_cuda();
    }

    let mut params = rnn.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.05, DType::F32).with_momentum(0.30);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before recurrent path training failed");
    }
    let start = Instant::now();
    for _ in 0..44 {
        let hidden = rnn.forward_step(&x, &h_prev);
        let pred = head.forward(hidden);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.recurrent");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.recurrent");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after recurrent path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn gru_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[3, 3],
        vec![0.15, -0.25, 0.30, 0.35, 0.05, -0.15, -0.10, 0.28, 0.22],
        dtype,
        device,
    );
    let h_prev = tensor_no_grad_with_dtype(
        &[3, 4],
        vec![
            0.08, -0.12, 0.18, 0.24, -0.18, 0.22, 0.12, -0.08, 0.26, 0.04, -0.16, 0.14,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[3, 2],
        vec![0.10, 0.20, 0.06, 0.18, 0.14, 0.12],
        dtype,
        device,
    );

    let gru = GRU::new_with_dtype(3, 4, dtype);
    for (idx, param) in gru.parameters().iter().enumerate() {
        init_parameter_pattern(param, dtype, 0.012, idx + 2, 0.001);
    }
    let head = Linear::new_with_dtype(4, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(
                IxDyn(&[2, 4]),
                vec![0.025, -0.015, 0.035, 0.020, -0.020, 0.045, 0.015, 0.030],
            )
            .expect("GRU path head weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.001),
        );
        head.bias
            .as_ref()
            .expect("GRU path head bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("GRU path head bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.001),
            );
    } else {
        init_linear(
            &head,
            vec![0.025, -0.015, 0.035, 0.020, -0.020, 0.045, 0.015, 0.030],
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        gru.to_cuda();
        head.to_cuda();
    }

    let mut params = gru.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.045, DType::F32).with_momentum(0.25);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before GRU path training failed");
    }
    let start = Instant::now();
    for _ in 0..46 {
        let hidden = gru.forward_step(&x, &h_prev);
        let pred = head.forward(hidden);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.gru");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.gru");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after GRU path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn lstm_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[3, 3],
        vec![0.18, -0.12, 0.32, -0.26, 0.18, 0.24, 0.12, 0.34, -0.16],
        dtype,
        device,
    );
    let h_prev = tensor_no_grad_with_dtype(
        &[3, 4],
        vec![
            0.06, -0.08, 0.16, 0.22, -0.14, 0.20, 0.08, -0.04, 0.24, 0.06, -0.18, 0.12,
        ],
        dtype,
        device,
    );
    let c_prev = tensor_no_grad_with_dtype(
        &[3, 4],
        vec![
            0.12, -0.16, 0.20, 0.08, -0.10, 0.18, 0.14, -0.06, 0.22, 0.04, -0.12, 0.16,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[3, 2],
        vec![0.11, 0.19, 0.07, 0.21, 0.15, 0.09],
        dtype,
        device,
    );

    let lstm = LSTM::new_with_dtype(3, 4, dtype);
    for (idx, param) in lstm.parameters().iter().enumerate() {
        init_parameter_pattern(param, dtype, 0.010, idx + 4, 0.001);
    }
    let head = Linear::new_with_dtype(4, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(
                IxDyn(&[2, 4]),
                vec![0.030, -0.010, 0.035, 0.018, -0.018, 0.040, 0.012, 0.028],
            )
            .expect("LSTM path head weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.001),
        );
        head.bias
            .as_ref()
            .expect("LSTM path head bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("LSTM path head bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.001),
            );
    } else {
        init_linear(
            &head,
            vec![0.030, -0.010, 0.035, 0.018, -0.018, 0.040, 0.012, 0.028],
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        lstm.to_cuda();
        head.to_cuda();
    }

    let mut params = lstm.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.040, DType::F32).with_momentum(0.25);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before LSTM path training failed");
    }
    let start = Instant::now();
    for _ in 0..48 {
        let (hidden, _cell) = lstm.forward_step(&x, &h_prev, &c_prev);
        let pred = head.forward(hidden);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.lstm");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.lstm");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after LSTM path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn adam_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[6, 3],
        vec![
            0.15, -0.20, 0.40, 0.35, 0.10, -0.25, -0.30, 0.45, 0.20, 0.55, -0.35, 0.05, -0.10,
            0.25, 0.50, 0.40, 0.30, -0.15,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[6, 2],
        vec![
            0.20, -0.12, 0.04, 0.18, 0.16, 0.02, -0.02, 0.24, 0.28, -0.05, 0.10, 0.20,
        ],
        dtype,
        device,
    );

    let head = Linear::new_with_dtype(3, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.02, -0.01, 0.03, -0.02, 0.01, 0.02])
                .expect("Adam path linear weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.002),
        );
        head.bias
            .as_ref()
            .expect("Adam path linear bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0])
                    .expect("Adam path linear bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.002),
            );
    } else {
        init_linear(
            &head,
            vec![0.02, -0.01, 0.03, -0.02, 0.01, 0.02],
            vec![0.0, 0.0],
            dtype,
        );
    }
    if device == Device::Cuda {
        head.to_cuda();
    }

    let params = head.parameters();
    let mut opt = Adam::new_with_dtype(params.clone(), 0.025, DType::F32);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before Adam path training failed");
    }
    let start = Instant::now();
    for _ in 0..48 {
        let pred = head.forward(x.clone());
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.adam");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_adam_state(&opt, params.len(), "path.train.adam");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after Adam path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn conv_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[2, 1, 3, 3],
        vec![
            0.10, 0.20, -0.10, 0.30, -0.20, 0.40, 0.05, 0.25, -0.15, -0.10, 0.15, 0.35, 0.20,
            -0.25, 0.10, 0.45, 0.05, -0.05,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(&[2, 2], vec![0.18, -0.08, 0.10, 0.16], dtype, device);

    let conv = Conv2D::new_with_dtype(1, 2, 2, 1, 0, dtype);
    init_conv2d(
        &conv,
        vec![0.12, -0.04, 0.08, 0.05, -0.06, 0.10, 0.03, 0.09],
        vec![0.01, -0.02],
        dtype,
    );
    let head = Linear::new_with_dtype(8, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(
                IxDyn(&[2, 8]),
                vec![
                    0.06, -0.02, 0.04, 0.05, -0.03, 0.07, 0.02, 0.01, -0.04, 0.05, 0.03, -0.02,
                    0.06, 0.01, -0.05, 0.04,
                ],
            )
            .expect("conv path head weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.003),
        );
        head.bias
            .as_ref()
            .expect("conv path head bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("conv path head bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.003),
            );
    } else {
        init_linear(
            &head,
            vec![
                0.06, -0.02, 0.04, 0.05, -0.03, 0.07, 0.02, 0.01, -0.04, 0.05, 0.03, -0.02, 0.06,
                0.01, -0.05, 0.04,
            ],
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        conv.to_cuda();
        head.to_cuda();
    }

    let mut params = conv.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.04, DType::F32).with_momentum(0.25);
    let relu = ReLU::new();
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before conv path training failed");
    }
    let start = Instant::now();
    for _ in 0..44 {
        let features = relu.forward(conv.forward(x.clone())).reshape(vec![2, 8]);
        let pred = head.forward(features);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.conv");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.conv");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after conv path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn conv_pool_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let x = tensor_no_grad_with_dtype(
        &[2, 1, 4, 4],
        vec![
            0.10, 0.30, -0.20, 0.05, 0.25, -0.15, 0.40, 0.20, -0.05, 0.35, 0.15, -0.25, 0.30, 0.10,
            -0.10, 0.45, -0.20, 0.05, 0.25, 0.35, 0.15, -0.30, 0.20, 0.10, 0.40, -0.10, 0.30,
            -0.05, 0.05, 0.25, -0.15, 0.20,
        ],
        dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(&[2, 2], vec![0.12, 0.22, 0.18, 0.08], dtype, device);

    let conv = Conv2D::new_with_dtype(1, 2, 3, 1, 1, dtype);
    init_conv2d(
        &conv,
        vec![
            0.06, -0.02, 0.04, 0.08, 0.03, -0.05, 0.02, 0.07, -0.01, -0.04, 0.05, 0.03, 0.01, 0.06,
            -0.02, 0.04, -0.03, 0.07,
        ],
        vec![0.01, -0.01],
        dtype,
    );
    let head = Linear::new_with_dtype(8, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(
                IxDyn(&[2, 8]),
                vec![
                    0.04, -0.02, 0.05, 0.03, -0.01, 0.06, 0.02, 0.04, 0.03, 0.05, -0.04, 0.02,
                    0.06, -0.01, 0.03, 0.05,
                ],
            )
            .expect("conv-pool path head weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.002),
        );
        head.bias
            .as_ref()
            .expect("conv-pool path head bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("conv-pool path head bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.002),
            );
    } else {
        init_linear(
            &head,
            vec![
                0.04, -0.02, 0.05, 0.03, -0.01, 0.06, 0.02, 0.04, 0.03, 0.05, -0.04, 0.02, 0.06,
                -0.01, 0.03, 0.05,
            ],
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        conv.to_cuda();
        head.to_cuda();
    }

    let mut params = conv.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.035, DType::F32).with_momentum(0.25);
    let relu = ReLU::new();
    let pool = MaxPool2D::new(2, 2);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before conv-pool path training failed");
    }
    let start = Instant::now();
    for _ in 0..46 {
        let conv_out = relu.forward(conv.forward(x.clone()));
        let pooled = pool.forward(conv_out).reshape(vec![2, 8]);
        let pred = head.forward(pooled);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.conv_pool");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.conv_pool");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after conv-pool path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn attention_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));
    let runtime_dtype = if dtype == DType::I8 {
        DType::BF16
    } else {
        dtype
    };

    let x = tensor_no_grad_with_dtype(
        &[2, 3, 8],
        vec![
            0.10, -0.20, 0.30, 0.05, -0.10, 0.25, 0.15, -0.05, 0.20, 0.10, -0.15, 0.35, 0.05,
            -0.25, 0.30, 0.12, -0.08, 0.22, 0.18, -0.12, 0.28, 0.04, -0.18, 0.26, 0.16, -0.06,
            0.24, 0.08, -0.14, 0.32, 0.06, -0.22, -0.18, 0.14, 0.28, 0.10, -0.04, 0.34, 0.02,
            -0.16, 0.12, -0.10, 0.36, 0.06, -0.20, 0.30, 0.18, -0.02,
        ],
        runtime_dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[6, 2],
        vec![
            0.12, -0.05, 0.08, 0.14, 0.16, -0.02, 0.04, 0.18, 0.20, 0.02, 0.10, 0.12,
        ],
        runtime_dtype,
        device,
    );

    let attn = SelfAttention::new_with_runtime_dtypes(
        8,
        2,
        1,
        3,
        10000.0,
        true,
        dtype,
        runtime_dtype,
        runtime_dtype,
    );
    init_linear_weight(&attn.w_q, patterned_values(64, 0.006, 0), dtype, 0.001);
    init_linear_weight(&attn.w_k, patterned_values(32, 0.007, 1), dtype, 0.001);
    init_linear_weight(&attn.w_v, patterned_values(32, 0.005, 2), dtype, 0.001);
    init_linear_weight(&attn.w_o, patterned_values(64, 0.006, 3), dtype, 0.001);

    let head = Linear::new_with_dtype(8, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(
                IxDyn(&[2, 8]),
                vec![
                    0.03, -0.01, 0.04, 0.02, -0.02, 0.05, 0.01, 0.03, -0.01, 0.04, 0.02, 0.05,
                    -0.03, 0.01, 0.04, 0.02,
                ],
            )
            .expect("attention path head weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.001),
        );
        head.bias
            .as_ref()
            .expect("attention path head bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("attention path head bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.001),
            );
    } else {
        init_linear(
            &head,
            vec![
                0.03, -0.01, 0.04, 0.02, -0.02, 0.05, 0.01, 0.03, -0.01, 0.04, 0.02, 0.05, -0.03,
                0.01, 0.04, 0.02,
            ],
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        attn.to_cuda();
        head.to_cuda();
    }

    let mut params = attn.parameters();
    params.extend(head.parameters());
    let mut opt = SGD::new_with_dtype(params.clone(), 0.04, DType::F32).with_momentum(0.20);
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before attention path training failed");
    }
    let start = Instant::now();
    for _ in 0..42 {
        let (attn_out, _) = attn.forward(x.clone(), None);
        let features = attn_out.reshape(vec![6, 8]);
        let pred = head.forward(features);
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.attention");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.attention");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after attention path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn transformer_block_training_path_stats(dtype: DType, device: Device) -> TrainingPathStats {
    let _cuda_enabled_guard =
        (device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard = (device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));
    let runtime_dtype = if dtype == DType::I8 {
        DType::BF16
    } else {
        dtype
    };

    let x = tensor_no_grad_with_dtype(
        &[2, 3, 8],
        vec![
            0.10, -0.20, 0.30, 0.05, -0.10, 0.25, 0.15, -0.05, 0.20, 0.10, -0.15, 0.35, 0.05,
            -0.25, 0.30, 0.12, -0.08, 0.22, 0.18, -0.12, 0.28, 0.04, -0.18, 0.26, 0.16, -0.06,
            0.24, 0.08, -0.14, 0.32, 0.06, -0.22, -0.18, 0.14, 0.28, 0.10, -0.04, 0.34, 0.02,
            -0.16, 0.12, -0.10, 0.36, 0.06, -0.20, 0.30, 0.18, -0.02,
        ],
        runtime_dtype,
        device,
    );
    let y = tensor_no_grad_with_dtype(
        &[6, 2],
        vec![
            0.10, 0.18, 0.06, 0.14, 0.16, 0.02, 0.04, 0.20, 0.18, -0.02, 0.12, 0.10,
        ],
        runtime_dtype,
        device,
    );

    let norm1 = RMSNorm::new_with_dtype(8, 1e-5, dtype);
    let norm2 = RMSNorm::new_with_dtype(8, 1e-5, dtype);
    for (phase, norm) in [(0usize, &norm1), (1usize, &norm2)] {
        let values = (0..8)
            .map(|idx| 0.96 + (((idx + phase) % 5) as f32) * 0.02)
            .collect::<Vec<_>>();
        if dtype == DType::I8 {
            norm.weight.set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[8]), values).expect("transformer norm shape"),
                ParameterQuantization::Int8.with_scale(0.01),
            );
        } else {
            norm.weight.set_array_f32_with_dtype(
                Array::from_shape_vec(IxDyn(&[8]), values).expect("transformer norm shape"),
                dtype,
            );
        }
    }

    let attn = SelfAttention::new_with_runtime_dtypes(
        8,
        2,
        1,
        3,
        10000.0,
        true,
        dtype,
        runtime_dtype,
        runtime_dtype,
    );
    init_linear_weight(&attn.w_q, patterned_values(64, 0.030, 0), dtype, 0.002);
    init_linear_weight(&attn.w_k, patterned_values(32, 0.028, 1), dtype, 0.002);
    init_linear_weight(&attn.w_v, patterned_values(32, 0.026, 2), dtype, 0.002);
    init_linear_weight(&attn.w_o, patterned_values(64, 0.024, 3), dtype, 0.002);

    let gate = Linear::new_no_bias_with_dtype(8, 12, dtype);
    let up = Linear::new_no_bias_with_dtype(8, 12, dtype);
    let down = Linear::new_no_bias_with_dtype(12, 8, dtype);
    init_linear_weight(&gate, patterned_values(96, 0.060, 4), dtype, 0.004);
    init_linear_weight(&up, patterned_values(96, 0.050, 5), dtype, 0.004);
    init_linear_weight(&down, patterned_values(96, 0.040, 6), dtype, 0.004);

    let head = Linear::new_with_dtype(8, 2, dtype);
    if dtype == DType::I8 {
        head.weight.set_array_f32_with_quantization(
            Array::from_shape_vec(
                IxDyn(&[2, 8]),
                vec![
                    0.03, -0.01, 0.04, 0.02, -0.02, 0.05, 0.01, 0.03, -0.01, 0.04, 0.02, 0.05,
                    -0.03, 0.01, 0.04, 0.02,
                ],
            )
            .expect("transformer path head weight shape mismatch"),
            ParameterQuantization::Int8.with_scale(0.001),
        );
        head.bias
            .as_ref()
            .expect("transformer path head bias")
            .set_array_f32_with_quantization(
                Array::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.01])
                    .expect("transformer path head bias shape mismatch"),
                ParameterQuantization::Int8.with_scale(0.001),
            );
    } else {
        init_linear(
            &head,
            vec![
                0.03, -0.01, 0.04, 0.02, -0.02, 0.05, 0.01, 0.03, -0.01, 0.04, 0.02, 0.05, -0.03,
                0.01, 0.04, 0.02,
            ],
            vec![0.0, 0.01],
            dtype,
        );
    }
    if device == Device::Cuda {
        norm1.to_cuda();
        attn.to_cuda();
        norm2.to_cuda();
        gate.to_cuda();
        up.to_cuda();
        down.to_cuda();
        head.to_cuda();
    }

    let mut params = norm1.parameters();
    params.extend(attn.parameters());
    params.extend(norm2.parameters());
    params.extend(gate.parameters());
    params.extend(up.parameters());
    params.extend(down.parameters());
    params.extend(head.parameters());
    let lr = if dtype == DType::I8 { 0.8 } else { 0.05 };
    let mut opt = SGD::new_with_dtype(params.clone(), lr, DType::F32).with_momentum(0.20);
    let silu = SiLU::new();
    let mut losses = Vec::new();

    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync before transformer-block path training failed");
    }
    let start = Instant::now();
    for _ in 0..40 {
        let normed = norm1.forward(x.clone());
        let (attn_out, _) = attn.forward(normed, None);
        let residual1 = x.clone() + attn_out;
        let ffn_in = norm2.forward(residual1.clone());
        let gate_act = silu.forward(gate.forward(ffn_in.clone()));
        let up_out = up.forward(ffn_in);
        let ffn_out = down.forward(gate_act * up_out);
        let block_out = residual1 + ffn_out;
        let pred = head.forward(block_out.reshape(vec![6, 8]));
        let loss = MSELoss::apply(&pred, &y);
        losses.push(scalar_value(&loss));
        loss.backward();
        if device == Device::Cuda {
            assert_cuda_f32_grads_without_host(&params, "path.train.transformer_block");
        }
        opt.step();
        if device == Device::Cuda {
            assert_cuda_f32_momentum_state(&opt, params.len(), "path.train.transformer_block");
        }
        opt.zero_grad();
    }
    if device == Device::Cuda {
        lumen::ops::cuda::synchronize()
            .expect("CUDA sync after transformer-block path training failed");
    }

    TrainingPathStats {
        losses,
        elapsed: start.elapsed(),
    }
}

fn count_loss_increases(losses: &[f32]) -> usize {
    losses.windows(2).filter(|pair| pair[1] > pair[0]).count()
}

fn assert_sgd_like_loss_trace(name: &str, losses: &[f32]) {
    assert_loss_trace_improves_below(name, losses, 0.05);
}

fn assert_loss_trace_improves_below(name: &str, losses: &[f32], best_fraction_of_first: f32) {
    assert!(!losses.is_empty(), "{name} produced no losses");
    assert!(
        losses.iter().all(|loss| loss.is_finite()),
        "{name} produced non-finite losses: {losses:?}"
    );
    let first = losses[0];
    let last = *losses.last().expect("checked non-empty");
    let best = losses
        .iter()
        .copied()
        .fold(f32::INFINITY, |acc, loss| acc.min(loss));
    assert!(
        last < first && best < first * best_fraction_of_first,
        "{name} loss trace does not look SGD-like: first={first:.6} last={last:.6} best={best:.6} required_best_below={:.6} losses={losses:?}",
        first * best_fraction_of_first
    );
}

fn run_training_path_check(args: &Args) {
    let cpu_stats = simple_training_path_stats(args.dtype, Device::Cpu);
    assert_sgd_like_loss_trace("path.train.cpu", &cpu_stats.losses);
    println!(
        "path.train.cpu             ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=SGD+momentum trace is not required to be monotonic",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = simple_training_path_stats(args.dtype, Device::Cuda);
        assert_sgd_like_loss_trace("path.train.cuda", &cuda_stats.losses);
        println!(
            "path.train.cuda            ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=lowp params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.cuda            skipped cuda_available=false");
    }
}

fn run_mlp_training_path_check(args: &Args) {
    let cpu_stats = mlp_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.70 } else { 0.05 };
    assert_loss_trace_improves_below("path.train.mlp.cpu", &cpu_stats.losses, required_fraction);
    println!(
        "path.train.mlp.cpu         ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Linear+ReLU+Linear with lowp data and f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = mlp_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.mlp.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.mlp.cuda        ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp parameters/data with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.mlp.cuda        skipped cuda_available=false");
    }
}

fn run_gelu_mlp_training_path_check(args: &Args) {
    let cpu_stats = gelu_mlp_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.85,
        DType::BF16 => 0.88,
        DType::I8 => 0.92,
    };
    assert_loss_trace_improves_below(
        "path.train.gelu_mlp.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.gelu_mlp.cpu    ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Linear+GELU+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = gelu_mlp_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.gelu_mlp.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.gelu_mlp.cuda   ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp GELU MLP params/data with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.gelu_mlp.cuda   skipped cuda_available=false");
    }
}

fn run_dropout_mlp_training_path_check(args: &Args) {
    let cpu_stats = dropout_mlp_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.82,
        DType::BF16 => 0.86,
        DType::I8 => 0.92,
    };
    assert_loss_trace_improves_below(
        "path.train.dropout.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.dropout.cpu    ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Linear+ReLU+Dropout+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = dropout_mlp_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.dropout.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.dropout.cuda   ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp Dropout MLP params/data with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.dropout.cuda   skipped cuda_available=false");
    }
}

fn run_batch_matmul_training_path_check(args: &Args) {
    let cpu_stats = batch_matmul_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.70,
        DType::BF16 => 0.76,
        DType::I8 => 0.84,
    };
    assert_loss_trace_improves_below(
        "path.train.batch_matmul.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.batch_matmul.cpu ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=BatchMatMul+MSE with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = batch_matmul_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.batch_matmul.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.batch_matmul.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp BatchMatMul params/data with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.batch_matmul.cuda skipped cuda_available=false");
    }
}

fn run_gated_mlp_training_path_check(args: &Args) {
    let cpu_stats = gated_mlp_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.92,
        DType::BF16 => 0.95,
        DType::I8 => 0.98,
    };
    assert_loss_trace_improves_below(
        "path.train.gated_mlp.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.gated_mlp.cpu   ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=SwiGLU-style Linear+SiLU+mul+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = gated_mlp_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.gated_mlp.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.gated_mlp.cuda  ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp gated-MLP params/data with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.gated_mlp.cuda  skipped cuda_available=false");
    }
}

fn run_residual_mix_training_path_check(args: &Args) {
    let cpu_stats = residual_mix_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.88,
        DType::BF16 => 0.92,
        DType::I8 => 0.99,
    };
    assert_loss_trace_improves_below(
        "path.train.residual_mix.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.residual_mix.cpu ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=residual Linear branch plus SiLU-gated multiplicative branch with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = residual_mix_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.residual_mix.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.residual_mix.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp residual/gated branches with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.residual_mix.cuda skipped cuda_available=false");
    }
}

fn run_broadcast_affine_training_path_check(args: &Args) {
    let cpu_stats = broadcast_affine_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.35,
        DType::BF16 => 0.40,
        DType::I8 => 0.45,
    };
    assert_loss_trace_improves_below(
        "path.train.broadcast_affine.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.broadcast_affine.cpu ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=row-broadcast scale/bias params with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = broadcast_affine_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.broadcast_affine.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.broadcast_affine.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp row-broadcast scale/bias params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.broadcast_affine.cuda skipped cuda_available=false");
    }
}

fn run_classifier_training_path_check(args: &Args) {
    let cpu_stats = classifier_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.98 } else { 0.95 };
    assert_loss_trace_improves_below(
        "path.train.classifier.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.classifier.cpu  ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Embedding+Linear+CrossEntropy with lowp params and f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = classifier_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.classifier.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.classifier.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp embedding/head with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.classifier.cuda skipped cuda_available=false");
    }
}

fn run_shape_training_path_check(args: &Args) {
    let cpu_stats = shape_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.70,
        DType::BF16 => 0.76,
        DType::I8 => 0.90,
    };
    assert_loss_trace_improves_below("path.train.shape.cpu", &cpu_stats.losses, required_fraction);
    println!(
        "path.train.shape.cpu       ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Linear+reshape+permute+cat+slice+MSE with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = shape_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.shape.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.shape.cuda      ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident shape/view chain with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.shape.cuda      skipped cuda_available=false");
    }
}

fn run_shared_parameter_training_path_check(args: &Args) {
    let cpu_stats = shared_parameter_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.55,
        DType::BF16 => 0.62,
        DType::I8 => 0.86,
    };
    assert_loss_trace_improves_below(
        "path.train.shared_param.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.shared_param.cpu ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=same lowp Linear parameter used twice in one graph with f32 accumulated gradients",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = shared_parameter_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.shared_param.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.shared_param.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident shared lowp params with accumulated f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.shared_param.cuda skipped cuda_available=false");
    }
}

fn run_gradient_accumulation_training_path_check(args: &Args) {
    let cpu_stats = gradient_accumulation_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.58,
        DType::BF16 => 0.64,
        DType::I8 => 0.90,
    };
    assert_loss_trace_improves_below(
        "path.train.grad_accum.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.grad_accum.cpu  ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=two micro-batch backward calls before one SGD step with f32 accumulated gradients",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = gradient_accumulation_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.grad_accum.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.grad_accum.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp params with f32 gradients accumulated across two CUDA backward calls",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.grad_accum.cuda skipped cuda_available=false");
    }
}

fn run_optimizer_batch_training_path_check(args: &Args) {
    let cpu_stats = optimizer_batch_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.88,
        DType::BF16 => 0.90,
        DType::I8 => 0.98,
    };
    assert_loss_trace_improves_below(
        "path.train.optimizer_batch.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.optimizer_batch.cpu ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=8-shard Linear fan-in sized to exercise batched optimizer updates",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = optimizer_batch_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.optimizer_batch.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.optimizer_batch.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=CUDA resident lowp shards with f32 grads/state and asserted batched optimizer fast path",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.optimizer_batch.cuda skipped cuda_available=false");
    }
}

fn run_adam_batch_training_path_check(args: &Args) {
    let cpu_stats = adam_batch_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.92,
        DType::BF16 => 0.94,
        DType::I8 => 1.02,
    };
    assert_loss_trace_improves_below(
        "path.train.adam_batch.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.adam_batch.cpu  ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=8-shard Linear fan-in sized to exercise batched Adam updates",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = adam_batch_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.adam_batch.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.adam_batch.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=CUDA resident lowp shards with f32 grads/state and asserted batched Adam fast path",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.adam_batch.cuda skipped cuda_available=false");
    }
}

fn run_norm_training_path_check(args: &Args) {
    let cpu_stats = norm_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.85 } else { 0.25 };
    assert_loss_trace_improves_below("path.train.norm.cpu", &cpu_stats.losses, required_fraction);
    println!(
        "path.train.norm.cpu        ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=RMSNorm+Linear with lowp data and f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = norm_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.norm.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.norm.cuda       ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp norm/head with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.norm.cuda       skipped cuda_available=false");
    }
}

fn run_recurrent_training_path_check(args: &Args) {
    let cpu_stats = recurrent_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.90 } else { 0.70 };
    assert_loss_trace_improves_below(
        "path.train.recurrent.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.recurrent.cpu   ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=RNN+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = recurrent_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.recurrent.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.recurrent.cuda  ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp recurrent/head params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.recurrent.cuda  skipped cuda_available=false");
    }
}

fn run_gru_training_path_check(args: &Args) {
    let cpu_stats = gru_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.92 } else { 0.75 };
    assert_loss_trace_improves_below("path.train.gru.cpu", &cpu_stats.losses, required_fraction);
    println!(
        "path.train.gru.cpu         ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=GRU+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = gru_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.gru.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.gru.cuda        ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp GRU/head params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.gru.cuda        skipped cuda_available=false");
    }
}

fn run_lstm_training_path_check(args: &Args) {
    let cpu_stats = lstm_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.92 } else { 0.75 };
    assert_loss_trace_improves_below("path.train.lstm.cpu", &cpu_stats.losses, required_fraction);
    println!(
        "path.train.lstm.cpu        ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=LSTM+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = lstm_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.lstm.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.lstm.cuda       ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp LSTM/head params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.lstm.cuda       skipped cuda_available=false");
    }
}

fn run_adam_training_path_check(args: &Args) {
    let cpu_stats = adam_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.80 } else { 0.10 };
    assert_loss_trace_improves_below("path.train.adam.cpu", &cpu_stats.losses, required_fraction);
    println!(
        "path.train.adam.cpu        ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Linear+MSE with Adam f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = adam_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.adam.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.adam.cuda       ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp params with f32 gradients and f32 Adam state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.adam.cuda       skipped cuda_available=false");
    }
}

fn run_conv_training_path_check(args: &Args) {
    let cpu_stats = conv_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.88 } else { 0.50 };
    assert_loss_trace_improves_below("path.train.conv.cpu", &cpu_stats.losses, required_fraction);
    println!(
        "path.train.conv.cpu        ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Conv2D+ReLU+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = conv_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.conv.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.conv.cuda       ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp conv/head with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.conv.cuda       skipped cuda_available=false");
    }
}

fn run_conv_pool_training_path_check(args: &Args) {
    let cpu_stats = conv_pool_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.92 } else { 0.70 };
    assert_loss_trace_improves_below(
        "path.train.conv_pool.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.conv_pool.cpu   ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=Conv2D+ReLU+MaxPool2D+Linear with f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step()
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = conv_pool_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.conv_pool.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.conv_pool.cuda  ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp conv/head with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.conv_pool.cuda  skipped cuda_available=false");
    }
}

fn run_attention_training_path_check(args: &Args) {
    let cpu_stats = attention_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = if args.dtype == DType::I8 { 0.95 } else { 0.90 };
    let runtime_dtype = if args.dtype == DType::I8 {
        DType::BF16
    } else {
        args.dtype
    };
    assert_loss_trace_improves_below(
        "path.train.attention.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.attention.cpu   ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=SelfAttention+Linear param_dtype={:?} runtime_dtype={:?} f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step(),
        args.dtype,
        runtime_dtype
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = attention_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.attention.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.attention.cuda  ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp attention/head params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.attention.cuda  skipped cuda_available=false");
    }
}

fn run_transformer_block_training_path_check(args: &Args) {
    let cpu_stats = transformer_block_training_path_stats(args.dtype, Device::Cpu);
    let required_fraction = match args.dtype {
        DType::F32 | DType::F16 => 0.90,
        DType::BF16 => 0.94,
        DType::I8 => 0.98,
    };
    let runtime_dtype = if args.dtype == DType::I8 {
        DType::BF16
    } else {
        args.dtype
    };
    assert_loss_trace_improves_below(
        "path.train.transformer_block.cpu",
        &cpu_stats.losses,
        required_fraction,
    );
    println!(
        "path.train.transformer.cpu ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=RMSNorm+SelfAttention+SwiGLU residual block param_dtype={:?} runtime_dtype={:?} f32 optimizer state",
        cpu_stats.first(),
        cpu_stats.last(),
        cpu_stats.best(),
        count_loss_increases(&cpu_stats.losses),
        cpu_stats.steps(),
        cpu_stats.elapsed.as_secs_f64() * 1e3,
        cpu_stats.us_per_step(),
        args.dtype,
        runtime_dtype
    );

    if lumen::ops::cuda::is_available() {
        let cuda_stats = transformer_block_training_path_stats(args.dtype, Device::Cuda);
        assert_loss_trace_improves_below(
            "path.train.transformer_block.cuda",
            &cuda_stats.losses,
            required_fraction,
        );
        println!(
            "path.train.transformer.cuda ok first={:.6} last={:.6} best={:.6} increases={} steps={} total_ms={:.3} us_per_step={:.2} note=resident lowp transformer block params with f32 gradients and f32 momentum state",
            cuda_stats.first(),
            cuda_stats.last(),
            cuda_stats.best(),
            count_loss_increases(&cuda_stats.losses),
            cuda_stats.steps(),
            cuda_stats.elapsed.as_secs_f64() * 1e3,
            cuda_stats.us_per_step()
        );
    } else {
        println!("path.train.transformer.cuda skipped cuda_available=false");
    }
}

fn load_real_llama_model(args: &Args, config: &LlamaConfig) -> LlamaModel {
    let weights = args
        .weights
        .as_ref()
        .expect("weights path should be checked before loading");
    let precision_config = PrecisionConfig {
        parameter_dtype: args.dtype,
        runtime_dtype: args.dtype,
        allow_parameter_dtype_copies: args.allow_parameter_copies,
    };
    let load_options = WeightLoadOptions {
        float_source_quantization: Default::default(),
        stream_from_disk: args.stream_weights,
    };
    let model = with_precision_config(precision_config, || {
        with_runtime_component_dtypes(Some(args.dtype), Some(args.dtype), || {
            with_parameter_init_mode(ParameterInitMode::Placeholder, || {
                LlamaModel::new(config.clone())
            })
        })
    });
    ModelLoader::load_llama_weights_with_options(weights, &model.named_parameters(), load_options)
        .expect("real-model inference path weight load failed");
    model
}

fn run_real_inference_path_check(args: &Args) {
    let Some(weights) = args.weights.as_ref() else {
        println!("path.infer.real            skipped missing --weights/--tokenizer");
        return;
    };
    let Some(tokenizer_path) = args.tokenizer.as_ref() else {
        println!("path.infer.real            skipped missing --weights/--tokenizer");
        return;
    };
    if !Path::new(weights).exists() {
        panic!("path.infer.real weights file does not exist: {weights}");
    }
    if !Path::new(tokenizer_path).exists() {
        panic!("path.infer.real tokenizer file does not exist: {tokenizer_path}");
    }
    if args.path_device == Device::Cuda && !lumen::ops::cuda::is_available() {
        println!("path.infer.real            skipped path_device=cuda cuda_available=false");
        return;
    }
    if args.dtype == DType::I8 {
        println!(
            "path.infer.real            skipped dtype=I8 note=this unified path uses one dtype for parameters/runtime/KV cache, and real Llama runtime currently requires floating dtypes"
        );
        return;
    }

    let tokenizer = LlamaTokenizer::from_file(tokenizer_path)
        .expect("real-model inference path tokenizer load failed");
    let config = real_llama_config(args.max_seq_len);
    assert_eq!(
        tokenizer.vocab_size(),
        config.vocab_size,
        "tokenizer/model vocab mismatch"
    );
    let prompt = build_first_turn_prompt(&args.system, &args.prompt);
    let prompt_tokens = tokenizer
        .encode(&prompt, false)
        .expect("real-model inference path tokenization failed");
    assert!(
        prompt_tokens.len() + args.max_gen + 2 < config.max_seq_len,
        "path.infer.real prompt_tokens={} max_gen={} exceed max_seq_len={}",
        prompt_tokens.len(),
        args.max_gen,
        config.max_seq_len
    );

    let _cuda_enabled_guard =
        (args.path_device == Device::Cuda).then(|| lumen::ops::cuda::set_enabled_scoped(true));
    let _strict_guard =
        (args.path_device == Device::Cuda).then(|| set_strict_device_execution_scoped(true));

    let load_start = Instant::now();
    let model = load_real_llama_model(args, &config);
    let load_elapsed = load_start.elapsed();
    let move_start = Instant::now();
    if args.path_device == Device::Cuda {
        model.to_cuda();
        lumen::ops::cuda::synchronize().expect("CUDA sync after real-model move failed");
    }
    let move_elapsed = move_start.elapsed();

    let stop_ids = generated_stop_ids(&tokenizer);
    let mut caches = model.init_kv_caches(1);
    model.reset_kv_caches(&mut caches);
    let mut generated = Vec::new();

    if args.path_device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync before real-model inference failed");
    }
    let infer_start = Instant::now();
    no_grad(|| {
        let prefill = token_ids_tensor(&prompt_tokens, args.path_device);
        let logits = model.forward_last_logits(prefill, &mut caches, 0);
        let mut next = argmax(&last_step_logits_vec(&logits));

        for _ in 0..args.max_gen {
            if stop_ids.contains(&next) {
                break;
            }
            generated.push(next);
            let decode = token_ids_tensor(&[next], args.path_device);
            let logits = model.forward_last_logits(decode, &mut caches, 0);
            next = argmax(&last_step_logits_vec(&logits));
            if caches[0].borrow().len + 2 >= config.max_seq_len {
                break;
            }
        }
    });
    if args.path_device == Device::Cuda {
        lumen::ops::cuda::synchronize().expect("CUDA sync after real-model inference failed");
    }
    let infer_elapsed = infer_start.elapsed();

    let text = tokenizer.decode(&generated, true);
    assert_inference_text_quality(&text);
    let tokens_per_second = if infer_elapsed.is_zero() {
        0.0
    } else {
        generated.len() as f64 / infer_elapsed.as_secs_f64()
    };
    println!(
        "path.infer.real            ok device={:?} prompt_tokens={} generated_tokens={} load_ms={:.3} move_ms={:.3} infer_ms={:.3} tok_s={:.2} chars={} replacement={} control={} repeat_run={}",
        args.path_device,
        prompt_tokens.len(),
        generated.len(),
        load_elapsed.as_secs_f64() * 1e3,
        move_elapsed.as_secs_f64() * 1e3,
        infer_elapsed.as_secs_f64() * 1e3,
        tokens_per_second,
        text.chars().count(),
        text.matches('\u{FFFD}').count(),
        text.chars()
            .filter(|ch| ch.is_control() && !ch.is_whitespace())
            .count(),
        max_repeated_char_run(&text)
    );
    if args.show_output {
        println!("path.infer.real.output:\n{}", text.trim());
    }
}

fn max_repeated_char_run(text: &str) -> usize {
    let mut prev = None;
    let mut cur = 0usize;
    let mut best = 0usize;
    for ch in text.chars() {
        if Some(ch) == prev {
            cur += 1;
        } else {
            prev = Some(ch);
            cur = 1;
        }
        best = best.max(cur);
    }
    best
}

fn assert_inference_text_quality(text: &str) {
    let chars = text.chars().collect::<Vec<_>>();
    let visible = chars.iter().filter(|ch| !ch.is_whitespace()).count();
    let replacement = chars.iter().filter(|&&ch| ch == '\u{FFFD}').count();
    let trailing_replacement = chars
        .iter()
        .rev()
        .take_while(|&&ch| ch == '\u{FFFD}')
        .count();
    let body_replacement = replacement.saturating_sub(trailing_replacement.min(4));
    let control = chars
        .iter()
        .filter(|&&ch| ch.is_control() && !ch.is_whitespace())
        .count();
    let bad_ratio = if chars.is_empty() {
        1.0
    } else {
        (body_replacement + control) as f32 / chars.len() as f32
    };

    assert!(
        visible >= 4,
        "inference path generated too little visible text: {text:?}"
    );
    assert!(
        body_replacement <= 1 && bad_ratio <= 0.05,
        "inference path generated likely garbled text: replacement={replacement} trailing_replacement={trailing_replacement} control={control} chars={} text={text:?}",
        chars.len()
    );
    assert!(
        max_repeated_char_run(text) <= 24,
        "inference path generated an excessive repeated-character run: {text:?}"
    );
}

pub(super) fn run_path_checks(args: &Args) {
    if !args.check {
        return;
    }
    let should_run_path = |name: &str| {
        args.case_filter.as_ref().is_none_or(|filter| {
            name == filter
                || name
                    .strip_prefix(filter)
                    .is_some_and(|suffix| suffix.starts_with('.'))
        })
    };
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train")
    {
        run_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.mlp")
    {
        run_mlp_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.gelu_mlp")
    {
        run_gelu_mlp_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.dropout")
    {
        run_dropout_mlp_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.batch_matmul")
    {
        run_batch_matmul_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.gated_mlp")
    {
        run_gated_mlp_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.residual_mix")
    {
        run_residual_mix_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.broadcast_affine")
    {
        run_broadcast_affine_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.classifier")
    {
        run_classifier_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.shape")
    {
        run_shape_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.shared_param")
    {
        run_shared_parameter_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.grad_accum")
    {
        run_gradient_accumulation_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.optimizer_batch")
    {
        run_optimizer_batch_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.adam_batch")
    {
        run_adam_batch_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.norm")
    {
        run_norm_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.recurrent")
    {
        run_recurrent_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.gru")
    {
        run_gru_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.lstm")
    {
        run_lstm_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.adam")
    {
        run_adam_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.conv")
    {
        run_conv_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.conv_pool")
    {
        run_conv_pool_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.attention")
    {
        run_attention_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Backward | Suite::Path)
        && should_run_path("path.train.transformer_block")
    {
        run_transformer_block_training_path_check(args);
    }
    if matches!(args.suite, Suite::All | Suite::Nn | Suite::Path)
        && should_run_path("path.infer.real")
    {
        run_real_inference_path_check(args);
    }
}
