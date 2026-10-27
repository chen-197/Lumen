//! Bounded-score F32 CPU inference; masks and K/V head groups remain views.
use super::{AttentionMask, AttentionOptions};
use crate::autograd::{Tensor, is_no_grad};
use crate::precision::DType;
use ndarray::linalg::general_mat_mul;
use ndarray::{Array4, ArrayView2, ArrayView4, ArrayViewMut2, Axis, Ix4, s};
use std::cell::RefCell;

// At most 256 KiB retained per executing worker. When one key row itself is
// larger, a temporary allocation is dropped after the job instead of retained.
const SCORE_VALUES: usize = 64 * 1024;
const QUERY_ROWS: usize = 128;
thread_local! {static SCORES: RefCell<Vec<f32>> = const {RefCell::new(Vec::new())};}

fn with_scores<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    SCORES.with(|slot| {
        if len <= SCORE_VALUES
            && let Ok(mut values) = slot.try_borrow_mut()
        {
            if values.capacity() < len {
                let extra = len - values.len();
                values.reserve_exact(extra);
            }
            values.resize(len, 0.0);
            f(&mut values)
        } else {
            f(&mut vec![0.0; len])
        }
    })
}

#[derive(Clone, Copy)]
enum Mask<'a> {
    None,
    Keep(ArrayView4<'a, bool>),
    Additive(ArrayView4<'a, f32>),
}

#[inline]
fn normalize(row: &mut [f32], end: usize, score: impl Fn(usize, f32) -> f32) {
    let (active, blocked) = row.split_at_mut(end);
    blocked.fill(0.0);
    let mut maximum = f32::NEG_INFINITY;
    for (j, s) in active.iter_mut().enumerate() {
        *s = score(j, *s);
        maximum = maximum.max(*s);
    }
    if maximum == f32::NEG_INFINITY {
        active.fill(0.0);
        return;
    }
    let mut sum = 0.0;
    for s in active.iter_mut() {
        *s = if *s == f32::NEG_INFINITY {
            0.0
        } else {
            (*s - maximum).exp()
        };
        sum += *s;
    }
    for s in active {
        *s /= sum;
    }
}

fn block(
    q: ArrayView2<'_, f32>,
    k: ArrayView2<'_, f32>,
    v: ArrayView2<'_, f32>,
    mask: Mask<'_>,
    batch: usize,
    head: usize,
    first: usize,
    scale: f32,
    options: AttentionOptions,
    mut output: ArrayViewMut2<'_, f32>,
) {
    let rows = q.nrows();
    let keys = if options.causal {
        k.nrows().min(
            options
                .query_offset
                .saturating_add(first)
                .saturating_add(rows),
        )
    } else {
        k.nrows()
    };
    let k = k.slice(s![..keys, ..]);
    let v = v.slice(s![..keys, ..]);
    with_scores(rows * keys, |scratch| {
        let mut scores = ndarray::ArrayViewMut2::from_shape((rows, keys), scratch).unwrap();
        general_mat_mul(1.0, &q, &k.t(), 0.0, &mut scores);
        for (i, mut row) in scores.outer_iter_mut().enumerate() {
            let query = first + i;
            let end = if options.causal {
                keys.min(options.query_offset.saturating_add(query).saturating_add(1))
            } else {
                keys
            };
            let row = row.as_slice_mut().unwrap();
            match mask {
                Mask::None => normalize(row, end, |_, s| s * scale),
                Mask::Keep(keep) => {
                    let keep = keep.slice(s![batch, head, query, ..]);
                    normalize(row, end, |j, s| {
                        if keep[j] {
                            s * scale
                        } else {
                            f32::NEG_INFINITY
                        }
                    });
                }
                Mask::Additive(bias) => {
                    let bias = bias.slice(s![batch, head, query, ..]);
                    normalize(row, end, |j, s| s * scale + bias[j]);
                }
            }
        }
        general_mat_mul(1.0, &scores, &v, 0.0, &mut output);
    });
}

pub(super) fn try_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&AttentionMask<'_>>,
    options: AttentionOptions,
) -> Option<Tensor> {
    if !query.is_cpu() || [query, key, value].iter().any(|t| t.dtype() != DType::F32) {
        return None;
    }
    let bias = match mask {
        Some(AttentionMask::Additive(bias)) => Some(*bias),
        _ => None,
    };
    if bias.is_some_and(|b| b.dtype() != DType::F32) {
        return None;
    }
    if !is_no_grad()
        && ([query, key, value].iter().any(|t| t.requires_grad())
            || bias.is_some_and(Tensor::requires_grad))
    {
        return None;
    }
    let qdata = query.data_ref();
    let kdata = key.data_ref();
    let vdata = value.data_ref();
    let q = qdata.view().into_dimensionality::<Ix4>().unwrap();
    let k = kdata.view().into_dimensionality::<Ix4>().unwrap();
    let v = vdata.view().into_dimensionality::<Ix4>().unwrap();
    let (batches, heads, queries, d) = q.dim();
    let keys = k.shape()[2];
    let dv = v.shape()[3];
    let n_rep = heads / k.shape()[1];
    let bias_data = bias.map(Tensor::data_ref);
    let mask = match mask {
        Some(AttentionMask::Keep(keep)) => {
            Mask::Keep(keep.broadcast((batches, heads, queries, keys)).unwrap())
        }
        Some(AttentionMask::Additive(_)) => Mask::Additive(
            bias_data
                .as_ref()
                .unwrap()
                .broadcast((batches, heads, queries, keys))
                .unwrap(),
        ),
        None => Mask::None,
    };
    let scale = options.scale.unwrap_or((d as f32).sqrt().recip());
    let maximum_rows = QUERY_ROWS.min((SCORE_VALUES / keys).max(1)).min(queries);
    let rows_per_job = queries.div_ceil(queries.div_ceil(maximum_rows));
    let mut output = Array4::<f32>::zeros((batches, heads, queries, dv));
    let mut jobs = Vec::new();
    let mut batches_out = output.view_mut();
    for bb in 0..batches {
        let (batch, tail) = batches_out.split_at(Axis(0), 1);
        batches_out = tail;
        let mut heads_out = batch.index_axis_move(Axis(0), 0);
        for hh in 0..heads {
            let (head, tail) = heads_out.split_at(Axis(0), 1);
            heads_out = tail;
            let mut remaining = head.index_axis_move(Axis(0), 0);
            for first in (0..queries).step_by(rows_per_job) {
                let rows = (queries - first).min(rows_per_job);
                let (part, tail) = remaining.split_at(Axis(0), rows);
                remaining = tail;
                jobs.push((bb, hh, first, part));
            }
        }
    }
    let work = rows_per_job
        .saturating_mul(keys)
        .saturating_mul(d.saturating_add(dv));
    crate::cpu::for_each_chunk_mut(&mut jobs, 1, work, |(_, job)| {
        let (bb, hh, first, part) = &mut job[0];
        let (bb, hh, first) = (*bb, *hh, *first);
        let rows = part.nrows();
        block(
            q.slice(s![bb, hh, first..first + rows, ..]),
            k.slice(s![bb, hh / n_rep, .., ..]),
            v.slice(s![bb, hh / n_rep, .., ..]),
            mask,
            bb,
            hh,
            first,
            scale,
            options,
            part.view_mut(),
        );
    });
    Some(Tensor::from_array_no_grad(output.into_dyn()))
}

#[cfg(test)]
pub(super) fn retained_score_capacity() -> usize {
    SCORES.with(|s| s.borrow().capacity())
}

#[cfg(test)]
#[path = "../../../tests/unit/layers/attention/cpu_infer/tests.rs"]
mod tests;
