use super::backward::{
    batch_matmul_backward_i8_slices, batch_matmul_backward_lhs_i8_slices,
    batch_matmul_backward_rhs_i8_slices, batch_matmul_backward_typed_slices,
    matmul_backward_lhs_i8_slices, matmul_backward_rhs_i8_slices,
};
use super::*;
use crate::accuracy_gate::{check, ordered, tensor, values};
use crate::autograd::Device;
use crate::cpu::CpuKernelContext;
use std::num::NonZeroUsize;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

struct Probe<'a> {
    caller: std::thread::ThreadId,
    pool: &'a rayon::ThreadPool,
    serial: bool,
    wrong: AtomicBool,
    calls: AtomicUsize,
    active: AtomicUsize,
    peak: AtomicUsize,
}

impl Probe<'_> {
    fn observe(&self) {
        let id = std::thread::current().id();
        let bad = if self.serial {
            id != self.caller
        } else {
            self.pool.current_thread_index().is_none()
        };
        self.wrong
            .fetch_or(bad || crate::cpu::parallelism() != 1, Ordering::SeqCst);
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        self.calls.fetch_add(1, Ordering::SeqCst);
        let mut nested = [0; 5];
        crate::cpu::for_each_chunk_mut(&mut nested, 1, 100_000, |(_, part)| {
            self.wrong
                .fetch_or(std::thread::current().id() != id, Ordering::SeqCst);
            part.fill(1);
        });
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy)]
struct Observed<'a> {
    probe: Option<&'a Probe<'a>>,
}
impl DotElem for Observed<'_> {
    fn to_f32(self) -> f32 {
        if let Some(probe) = self.probe {
            probe.observe();
        }
        1.
    }
}

#[test]
fn native_backward_workers_obey_pool_budget_grain_and_nesting() {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    for budget in [1, 2, 9] {
        let context = CpuKernelContext::with_pool(pool.clone())
            .with_task_limit(NonZeroUsize::new(budget).unwrap());
        for (m, k, n) in [(3, 5, 7), (37, 129, 67)] {
            let execute = |nested: bool| {
                let probe = Probe {
                    caller: std::thread::current().id(),
                    pool: &pool,
                    serial: nested || budget == 1 || m * k * n < 65536,
                    wrong: AtomicBool::new(false),
                    calls: AtomicUsize::new(0),
                    active: AtomicUsize::new(0),
                    peak: AtomicUsize::new(0),
                };
                let observed = |len| {
                    let mut v = vec![Observed { probe: None }; len];
                    v[0].probe = Some(&probe);
                    v
                };
                let a = observed(m * k);
                let b = observed(n * k);
                let g = vec![0.5; m * n];
                let ai8 = vec![1; m * k];
                let bi8 = vec![1; n * k];
                for (da, db, ea, eb) in [
                    {
                        let (da, db) = matmul_backward_rhs_i8_slices(&a, &bi8, 0.25, &g, m, k, n);
                        (da, db, 0.125 * n as f32, 0.5 * m as f32)
                    },
                    {
                        let (da, db) = matmul_backward_lhs_i8_slices(&ai8, 0.25, &b, &g, m, k, n);
                        (da, db, 0.5 * n as f32, 0.125 * m as f32)
                    },
                    {
                        let (da, db) = batch_matmul_backward_typed_slices(&a, &b, &g, 1, m, k, n);
                        (da, db, 0.5 * n as f32, 0.5 * m as f32)
                    },
                    {
                        let (da, db) =
                            batch_matmul_backward_rhs_i8_slices(&a, &bi8, 0.25, &g, 1, m, k, n);
                        (da, db, 0.125 * n as f32, 0.5 * m as f32)
                    },
                    {
                        let (da, db) =
                            batch_matmul_backward_lhs_i8_slices(&ai8, 0.25, &b, &g, 1, m, k, n);
                        (da, db, 0.5 * n as f32, 0.125 * m as f32)
                    },
                ] {
                    assert!(da.iter().all(|&x| x == ea));
                    assert!(db.iter().all(|&x| x == eb));
                }
                assert!(probe.calls.load(Ordering::SeqCst) > 0);
                assert!(
                    !probe.wrong.load(Ordering::SeqCst),
                    "budget={budget}, nested={nested}, shape={m}/{k}/{n}"
                );
                assert!(
                    probe.peak.load(Ordering::SeqCst)
                        <= if probe.serial { 1 } else { budget.min(3) }
                );
            };
            context.scope(|| execute(false));
            // Exercise the real slice VJPs from a numeric worker without moving a Tensor.
            context.scope(|| {
                crate::cpu::for_each_chunk_mut(&mut [0; 3], 1, 100_000, |(_, _)| execute(true))
            });
        }
    }
}

fn data(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + seed) % 251) as f32 - 125.) * 0.00371)
        .collect()
}

#[test]
fn contiguous_bmm_weight_updates_preserve_scalar_reduction_bits() {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let (b, k) = (2, 3);
    let scale = 0.00371f32;
    for m in [0, 1, 3, 17, 65] {
        for n in [0, 1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 129, 257] {
            let lhs = (0..b * m * k)
                .map(|i| ((i % 7) as f32 - 3.) * 0.625)
                .collect::<Vec<_>>();
            let ints = (0..b * m * k)
                .map(|i| ((i % 251) as i16 - 125) as i8)
                .collect::<Vec<_>>();
            let half = lhs.iter().copied().map(f16::from_f32).collect::<Vec<_>>();
            let bf = lhs.iter().copied().map(bf16::from_f32).collect::<Vec<_>>();
            let grad = (0..b * m * n)
                .map(|i| [0., -0., 1.125, -1.125, 262144., -262144., 0.00001][i % 7])
                .collect::<Vec<_>>();
            let rhs = vec![f16::ONE; b * k * n];
            let rhs_bf = vec![bf16::ONE; b * k * n];
            let rhs_float = vec![1.; b * k * n];
            let rhs_i8 = vec![1; b * k * n];
            // Independent column-major scalar traversal of the original formula.
            let reference = |decoded: &[f32]| {
                let mut result = vec![0f32; b * k * n];
                for bh in 0..b {
                    for kk in 0..k {
                        for col in 0..n {
                            let mut acc = 0f32;
                            for row in 0..m {
                                acc += decoded[(bh * m + row) * k + kk]
                                    * grad[(bh * m + row) * n + col];
                            }
                            result[(bh * k + kk) * n + col] = acc;
                        }
                    }
                }
                result
            };
            let expected = reference(&lhs);
            let quantized = reference(&ints.iter().map(|&v| v as f32 * scale).collect::<Vec<_>>());
            for budget in [1, 3] {
                let context = CpuKernelContext::with_pool(pool.clone())
                    .with_task_limit(NonZeroUsize::new(budget).unwrap());
                let outputs = context.scope(|| {
                    [
                        batch_matmul_backward_typed_slices(&half, &rhs, &grad, b, m, k, n).1,
                        batch_matmul_backward_typed_slices(&bf, &rhs_bf, &grad, b, m, k, n).1,
                        batch_matmul_backward_rhs_i8_slices(
                            &lhs, &rhs_i8, scale, &grad, b, m, k, n,
                        )
                        .1,
                        batch_matmul_backward_rhs_i8_slices(
                            &half, &rhs_i8, scale, &grad, b, m, k, n,
                        )
                        .1,
                        batch_matmul_backward_rhs_i8_slices(&bf, &rhs_i8, scale, &grad, b, m, k, n)
                            .1,
                        batch_matmul_backward_i8_slices(
                            &ints, scale, &rhs_i8, scale, &grad, b, m, k, n,
                        )
                        .1,
                        batch_matmul_backward_lhs_i8_slices(
                            &ints, scale, &rhs_float, &grad, b, m, k, n,
                        )
                        .1,
                        batch_matmul_backward_lhs_i8_slices(&ints, scale, &rhs, &grad, b, m, k, n)
                            .1,
                        batch_matmul_backward_lhs_i8_slices(
                            &ints, scale, &rhs_bf, &grad, b, m, k, n,
                        )
                        .1,
                    ]
                });
                for (index, output) in outputs.iter().enumerate() {
                    let expected = if index < 5 { &expected } else { &quantized };
                    assert_eq!(output.len(), expected.len());
                    for (&actual, &expected) in output.iter().zip(expected) {
                        assert_eq!(
                            actual.to_bits(),
                            expected.to_bits(),
                            "m={m}, n={n}, path={index}, budget={budget}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bmm_all_dtype_pairs_strided_gradients_match_f64_under_budgets() {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap(),
    );
    let types = [DType::F32, DType::F16, DType::BF16, DType::I8];
    for xd in types {
        for wd in types {
            for layout in 0..3 {
                for (case, (batch, heads, m, k, n)) in
                    [(1, 1, 3, 5, 7), (1, 2, 17, 65, 67), (2, 3, 9, 17, 19)]
                        .into_iter()
                        .enumerate()
                {
                    let shape = [batch, heads, m, k];
                    let wshape = [batch, heads, k, n];
                    let oshape = [batch, heads, m, n];
                    let x = tensor(
                        &shape,
                        &data(batch * heads * m * k, 3),
                        xd,
                        layout,
                        Device::Cpu,
                    );
                    let w = tensor(
                        &wshape,
                        &data(batch * heads * k * n, 7),
                        wd,
                        layout,
                        Device::Cpu,
                    );
                    let xv = values(&x);
                    let wv = values(&w);
                    let dy = data(batch * heads * m * n, 11);
                    let mut dx = vec![0f64; xv.len()];
                    let mut dw = vec![0f64; wv.len()];
                    for bh in 0..batch * heads {
                        for row in 0..m {
                            for kk in 0..k {
                                for col in 0..n {
                                    let xi = (bh * m + row) * k + kk;
                                    let wi = (bh * k + kk) * n + col;
                                    let gi = (bh * m + row) * n + col;
                                    dx[xi] += dy[gi] as f64 * wv[wi] as f64;
                                    dw[wi] += xv[xi] as f64 * dy[gi] as f64;
                                }
                            }
                        }
                    }
                    for budget in [1, 2, 9] {
                        x.zero_grad();
                        w.zero_grad();
                        let context = CpuKernelContext::with_pool(pool.clone())
                            .with_task_limit(NonZeroUsize::new(budget).unwrap());
                        context.scope(|| {
                            batch_matmul(&x, &w).backward_with_grad(ordered(&oshape, &dy, layout))
                        });
                        let gx = x.grad().unwrap().iter().copied().collect::<Vec<_>>();
                        let gw = w.grad().unwrap().iter().copied().collect::<Vec<_>>();
                        check("budgeted BMM input VJP", &gx, &dx, 8e-6, 8e-6);
                        check("budgeted BMM weight VJP", &gw, &dw, 8e-6, 8e-6);
                        if budget == 9 && std::env::var_os("LUMEN_OPERATOR_AUDIT").is_some() {
                            println!(
                                "BMM_BACKWARD_AUDIT={}",
                                serde_json::json!({"case":case,"layout":layout,"input_dtype":format!("{xd:?}"),"weight_dtype":format!("{wd:?}"),"x":{"shape":shape,"values":xv,"grad":gx},"w":{"shape":wshape,"values":wv,"grad":gw},"dy":dy})
                            );
                        }
                    }
                }
            }
        }
    }
}
