use crate::autograd::{Tensor, TensorData, is_no_grad};
use ndarray::{Array2, Array4, Ix2, Ix4, IxDyn, Zip};
use ndarray::linalg::general_mat_mul;
use rayon::prelude::*;
use half::bf16;
use std::cell::RefCell;
use std::rc::Rc;

const MATVEC_BLOCK_ROWS: usize = 16;
const ARGMAX_BLOCK_ROWS: usize = 32;
const MATVEC_BLOCK_THRESHOLD: usize = 16384;
const MATVEC_PAR_THRESHOLD: usize = 256;


const BF16_PREFILL_TOKEN_TILE: usize = 4;
const BF16_PREFILL_OUT_TILE: usize = 2;

#[inline]
pub fn matmul_prefill_rowmajor_bf16(
    x_rowmajor: &[f32],
    w_rowmajor: &[bf16],
    m_dim: usize,
    n_rows: usize,
    k_dim: usize,
    out_rowmajor: &mut [f32],
    bias: Option<&[f32]>,
) {
    assert_eq!(x_rowmajor.len(), m_dim * k_dim, "x size mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out_rowmajor.len(), m_dim * n_rows, "out size mismatch");
    if let Some(bias) = bias {
        assert_eq!(bias.len(), n_rows, "bias size mismatch");
    }

    let w_bits = bf16_slice_as_u16(w_rowmajor);
    let out_tile_len = BF16_PREFILL_TOKEN_TILE * n_rows;
    let x_tile_len = BF16_PREFILL_TOKEN_TILE * k_dim;

    out_rowmajor
        .par_chunks_mut(out_tile_len)
        .zip(x_rowmajor.par_chunks(x_tile_len))
        .for_each(|(out_chunk, x_chunk)| {
            let rows = x_chunk.len() / k_dim;
            debug_assert_eq!(out_chunk.len(), rows * n_rows);

            match bias {
                Some(bias) => {
                    for t in 0..rows {
                        let dst = &mut out_chunk[t * n_rows..(t + 1) * n_rows];
                        dst.copy_from_slice(bias);
                    }
                }
                None => out_chunk.fill(0.0),
            }

            let mut row_start = 0usize;
            while row_start + BF16_PREFILL_OUT_TILE <= n_rows {
                let row0 = &w_bits[row_start * k_dim..(row_start + 1) * k_dim];
                let row1 = &w_bits[(row_start + 1) * k_dim..(row_start + 2) * k_dim];

                let mut acc00 = 0.0f32;
                let mut acc01 = 0.0f32;
                let mut acc10 = 0.0f32;
                let mut acc11 = 0.0f32;
                let mut acc20 = 0.0f32;
                let mut acc21 = 0.0f32;
                let mut acc30 = 0.0f32;
                let mut acc31 = 0.0f32;

                let mut kk = 0usize;
                while kk + 4 <= k_dim {
                    let w00 = bf16_bits_to_f32(row0[kk]);
                    let w01 = bf16_bits_to_f32(row1[kk]);
                    let w10 = bf16_bits_to_f32(row0[kk + 1]);
                    let w11 = bf16_bits_to_f32(row1[kk + 1]);
                    let w20 = bf16_bits_to_f32(row0[kk + 2]);
                    let w21 = bf16_bits_to_f32(row1[kk + 2]);
                    let w30 = bf16_bits_to_f32(row0[kk + 3]);
                    let w31 = bf16_bits_to_f32(row1[kk + 3]);

                    if rows > 0 {
                        let x0 = x_chunk[kk];
                        let x1 = x_chunk[kk + 1];
                        let x2 = x_chunk[kk + 2];
                        let x3 = x_chunk[kk + 3];
                        acc00 += x0 * w00 + x1 * w10 + x2 * w20 + x3 * w30;
                        acc01 += x0 * w01 + x1 * w11 + x2 * w21 + x3 * w31;
                    }
                    if rows > 1 {
                        let base = k_dim + kk;
                        let x0 = x_chunk[base];
                        let x1 = x_chunk[base + 1];
                        let x2 = x_chunk[base + 2];
                        let x3 = x_chunk[base + 3];
                        acc10 += x0 * w00 + x1 * w10 + x2 * w20 + x3 * w30;
                        acc11 += x0 * w01 + x1 * w11 + x2 * w21 + x3 * w31;
                    }
                    if rows > 2 {
                        let base = 2 * k_dim + kk;
                        let x0 = x_chunk[base];
                        let x1 = x_chunk[base + 1];
                        let x2 = x_chunk[base + 2];
                        let x3 = x_chunk[base + 3];
                        acc20 += x0 * w00 + x1 * w10 + x2 * w20 + x3 * w30;
                        acc21 += x0 * w01 + x1 * w11 + x2 * w21 + x3 * w31;
                    }
                    if rows > 3 {
                        let base = 3 * k_dim + kk;
                        let x0 = x_chunk[base];
                        let x1 = x_chunk[base + 1];
                        let x2 = x_chunk[base + 2];
                        let x3 = x_chunk[base + 3];
                        acc30 += x0 * w00 + x1 * w10 + x2 * w20 + x3 * w30;
                        acc31 += x0 * w01 + x1 * w11 + x2 * w21 + x3 * w31;
                    }
                    kk += 4;
                }

                while kk < k_dim {
                    let w0 = bf16_bits_to_f32(row0[kk]);
                    let w1 = bf16_bits_to_f32(row1[kk]);
                    if rows > 0 {
                        let x = x_chunk[kk];
                        acc00 += x * w0;
                        acc01 += x * w1;
                    }
                    if rows > 1 {
                        let x = x_chunk[k_dim + kk];
                        acc10 += x * w0;
                        acc11 += x * w1;
                    }
                    if rows > 2 {
                        let x = x_chunk[2 * k_dim + kk];
                        acc20 += x * w0;
                        acc21 += x * w1;
                    }
                    if rows > 3 {
                        let x = x_chunk[3 * k_dim + kk];
                        acc30 += x * w0;
                        acc31 += x * w1;
                    }
                    kk += 1;
                }

                if rows > 0 {
                    let dst = &mut out_chunk[row_start..row_start + 2];
                    dst[0] += acc00;
                    dst[1] += acc01;
                }
                if rows > 1 {
                    let base = n_rows + row_start;
                    let dst = &mut out_chunk[base..base + 2];
                    dst[0] += acc10;
                    dst[1] += acc11;
                }
                if rows > 2 {
                    let base = 2 * n_rows + row_start;
                    let dst = &mut out_chunk[base..base + 2];
                    dst[0] += acc20;
                    dst[1] += acc21;
                }
                if rows > 3 {
                    let base = 3 * n_rows + row_start;
                    let dst = &mut out_chunk[base..base + 2];
                    dst[0] += acc30;
                    dst[1] += acc31;
                }

                row_start += BF16_PREFILL_OUT_TILE;
            }

            while row_start < n_rows {
                let row = &w_rowmajor[row_start * k_dim..(row_start + 1) * k_dim];
                for t in 0..rows {
                    let x = &x_chunk[t * k_dim..(t + 1) * k_dim];
                    out_chunk[t * n_rows + row_start] += dot_unrolled_bf16(x, row);
                }
                row_start += 1;
            }
        });
}

#[inline]
fn bf16_slice_as_u16(x: &[bf16]) -> &[u16] {
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u16, x.len()) }
}

#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    bf16::from_bits(bits).to_f32()
}

#[inline]
fn dot_unrolled_bf16_bits(x: &[f32], row: &[u16]) -> f32 {
    let mut a = 0.0f32;
    let mut b = 0.0f32;
    let mut c = 0.0f32;
    let mut d = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        a += bf16_bits_to_f32(row[kk]) * x0 + bf16_bits_to_f32(row[kk + 4]) * x4;
        b += bf16_bits_to_f32(row[kk + 1]) * x1 + bf16_bits_to_f32(row[kk + 5]) * x5;
        c += bf16_bits_to_f32(row[kk + 2]) * x2 + bf16_bits_to_f32(row[kk + 6]) * x6;
        d += bf16_bits_to_f32(row[kk + 3]) * x3 + bf16_bits_to_f32(row[kk + 7]) * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        a += bf16_bits_to_f32(row[kk]) * x[kk];
        b += bf16_bits_to_f32(row[kk + 1]) * x[kk + 1];
        c += bf16_bits_to_f32(row[kk + 2]) * x[kk + 2];
        d += bf16_bits_to_f32(row[kk + 3]) * x[kk + 3];
        kk += 4;
    }

    let mut sum = a + b + c + d;
    while kk < k_dim {
        sum += bf16_bits_to_f32(row[kk]) * x[kk];
        kk += 1;
    }
    sum
}


#[inline]
pub(crate) fn dot_unrolled(x: &[f32], row: &[f32]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        s0 += row[kk] * x[kk] + row[kk + 4] * x[kk + 4];
        s1 += row[kk + 1] * x[kk + 1] + row[kk + 5] * x[kk + 5];
        s2 += row[kk + 2] * x[kk + 2] + row[kk + 6] * x[kk + 6];
        s3 += row[kk + 3] * x[kk + 3] + row[kk + 7] * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk] * x[kk];
        s1 += row[kk + 1] * x[kk + 1];
        s2 += row[kk + 2] * x[kk + 2];
        s3 += row[kk + 3] * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk] * x[kk];
        kk += 1;
    }
    sum
}

#[inline]
pub(crate) fn matvec_rowmajor_serial(x: &[f32], w_rowmajor: &[f32], n_rows: usize, k_dim: usize, out: &mut [f32]) {
    for i in 0..n_rows {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        out[i] = dot_unrolled(x, row);
    }
}

#[inline]
fn matvec_rowmajor_rowwise_parallel(x: &[f32], w_rowmajor: &[f32], n_rows: usize, k_dim: usize, out: &mut [f32]) {
    out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        *out_val = dot_unrolled(x, row);
    });
}

#[inline]
fn matvec_rowmajor_block_parallel(x: &[f32], w_rowmajor: &[f32], n_rows: usize, k_dim: usize, out: &mut [f32]) {
    out.par_chunks_mut(MATVEC_BLOCK_ROWS)
        .enumerate()
        .for_each(|(block_idx, out_chunk)| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = out_chunk.len();
            let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut acc = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    acc[r] += w_block[base] * x0
                        + w_block[base + 1] * x1
                        + w_block[base + 2] * x2
                        + w_block[base + 3] * x3
                        + w_block[base + 4] * x4
                        + w_block[base + 5] * x5
                        + w_block[base + 6] * x6
                        + w_block[base + 7] * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for r in 0..rows {
                    acc[r] += w_block[r * k_dim + kk] * xv;
                }
                kk += 1;
            }

            out_chunk.copy_from_slice(&acc[..rows]);
        });
}

#[inline]
pub fn matvec_rowmajor_parallel(x: &[f32], w_rowmajor: &[f32], n_rows: usize, k_dim: usize, out: &mut [f32]) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        matvec_rowmajor_serial(x, w_rowmajor, n_rows, k_dim, out);
    } else if n_rows >= MATVEC_BLOCK_THRESHOLD {
        matvec_rowmajor_block_parallel(x, w_rowmajor, n_rows, k_dim, out);
    } else {
        matvec_rowmajor_rowwise_parallel(x, w_rowmajor, n_rows, k_dim, out);
    }
}


#[inline]
pub(crate) fn dot2_unrolled(x: &[f32], row0: &[f32], row1: &[f32]) -> (f32, f32) {
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk] * x0 + row0[kk + 4] * x4;
        a1 += row1[kk] * x0 + row1[kk + 4] * x4;
        b0 += row0[kk + 1] * x1 + row0[kk + 5] * x5;
        b1 += row1[kk + 1] * x1 + row1[kk + 5] * x5;
        c0 += row0[kk + 2] * x2 + row0[kk + 6] * x6;
        c1 += row1[kk + 2] * x2 + row1[kk + 6] * x6;
        d0 += row0[kk + 3] * x3 + row0[kk + 7] * x7;
        d1 += row1[kk + 3] * x3 + row1[kk + 7] * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] * x0;
        a1 += row1[kk] * x0;
        b0 += row0[kk + 1] * x1;
        b1 += row1[kk + 1] * x1;
        c0 += row0[kk + 2] * x2;
        c1 += row1[kk + 2] * x2;
        d0 += row0[kk + 3] * x3;
        d1 += row1[kk + 3] * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[inline]
pub(crate) fn dual_matvec_rowmajor_parallel(
    x: &[f32],
    w0_rowmajor: &[f32],
    w1_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled(x, row0, row1);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        out0.par_iter_mut()
            .zip(out1.par_iter_mut())
            .enumerate()
            .for_each(|(i, (dst0, dst1))| {
                let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (s0, s1) = dot2_unrolled(x, row0, row1);
                *dst0 = s0;
                *dst1 = s1;
            });
    }
}

#[inline]
pub fn dual_matvec_silu_mul_rowmajor_parallel(
    x: &[f32],
    gate_w_rowmajor: &[f32],
    up_w_rowmajor: &[f32],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(gate_w_rowmajor.len(), n_rows * k_dim, "gate weight size mismatch");
    assert_eq!(up_w_rowmajor.len(), n_rows * k_dim, "up weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            *out_val = (g * sig) * u;
        });
    }
}

#[inline]
pub fn matvec_argmax_rowmajor_parallel(x: &[f32], w_rowmajor: &[f32], n_rows: usize, k_dim: usize) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if n_rows >= MATVEC_BLOCK_THRESHOLD {
        let n_blocks = (n_rows + ARGMAX_BLOCK_ROWS - 1) / ARGMAX_BLOCK_ROWS;
        return (0..n_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let row_start = block_idx * ARGMAX_BLOCK_ROWS;
                let rows = (n_rows - row_start).min(ARGMAX_BLOCK_ROWS);
                let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
                let mut acc = [0.0f32; ARGMAX_BLOCK_ROWS];

                let mut kk = 0usize;
                while kk + 8 <= k_dim {
                    let x0 = x[kk];
                    let x1 = x[kk + 1];
                    let x2 = x[kk + 2];
                    let x3 = x[kk + 3];
                    let x4 = x[kk + 4];
                    let x5 = x[kk + 5];
                    let x6 = x[kk + 6];
                    let x7 = x[kk + 7];
                    for r in 0..rows {
                        let base = r * k_dim + kk;
                        acc[r] += w_block[base] * x0
                            + w_block[base + 1] * x1
                            + w_block[base + 2] * x2
                            + w_block[base + 3] * x3
                            + w_block[base + 4] * x4
                            + w_block[base + 5] * x5
                            + w_block[base + 6] * x6
                            + w_block[base + 7] * x7;
                    }
                    kk += 8;
                }

                while kk < k_dim {
                    let xv = x[kk];
                    for r in 0..rows {
                        acc[r] += w_block[r * k_dim + kk] * xv;
                    }
                    kk += 1;
                }

                let mut best = (row_start, f32::NEG_INFINITY);
                for r in 0..rows {
                    let cand = (row_start + r, acc[r]);
                    if cand.1 > best.1 {
                        best = cand;
                    }
                }
                best
            })
            .reduce(|| (0usize, f32::NEG_INFINITY), |a, b| if a.1 >= b.1 { a } else { b })
            .0;
    }

    if n_rows < MATVEC_PAR_THRESHOLD {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled(x, row);
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    } else {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled(x, row))
            })
            .reduce(|| (0usize, f32::NEG_INFINITY), |a, b| if a.1 >= b.1 { a } else { b })
            .0
    }
}


#[inline]
fn dot_unrolled_bf16(x: &[f32], row: &[bf16]) -> f32 {
    dot_unrolled_bf16_bits(x, bf16_slice_as_u16(row))
}


#[inline]
pub(crate) fn matvec_rowmajor_serial_bf16(x: &[f32], w_rowmajor: &[bf16], n_rows: usize, k_dim: usize, out: &mut [f32]) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    let mut i = 0usize;
    while i + 1 < n_rows {
        let row0 = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        let row1 = &w_rowmajor[(i + 1) * k_dim..(i + 2) * k_dim];
        let (s0, s1) = dot2_unrolled_bf16(x, row0, row1);
        out[i] = s0;
        out[i + 1] = s1;
        i += 2;
    }

    if i < n_rows {
        let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
        out[i] = dot_unrolled_bf16(x, row);
    }
}

#[inline]
pub(crate) fn dot2_unrolled_bf16(x: &[f32], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    let row0 = bf16_slice_as_u16(row0);
    let row1 = bf16_slice_as_u16(row1);
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += bf16_bits_to_f32(row0[kk]) * x0 + bf16_bits_to_f32(row0[kk + 4]) * x4;
        a1 += bf16_bits_to_f32(row1[kk]) * x0 + bf16_bits_to_f32(row1[kk + 4]) * x4;
        b0 += bf16_bits_to_f32(row0[kk + 1]) * x1 + bf16_bits_to_f32(row0[kk + 5]) * x5;
        b1 += bf16_bits_to_f32(row1[kk + 1]) * x1 + bf16_bits_to_f32(row1[kk + 5]) * x5;
        c0 += bf16_bits_to_f32(row0[kk + 2]) * x2 + bf16_bits_to_f32(row0[kk + 6]) * x6;
        c1 += bf16_bits_to_f32(row1[kk + 2]) * x2 + bf16_bits_to_f32(row1[kk + 6]) * x6;
        d0 += bf16_bits_to_f32(row0[kk + 3]) * x3 + bf16_bits_to_f32(row0[kk + 7]) * x7;
        d1 += bf16_bits_to_f32(row1[kk + 3]) * x3 + bf16_bits_to_f32(row1[kk + 7]) * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += bf16_bits_to_f32(row0[kk]) * x0;
        a1 += bf16_bits_to_f32(row1[kk]) * x0;
        b0 += bf16_bits_to_f32(row0[kk + 1]) * x1;
        b1 += bf16_bits_to_f32(row1[kk + 1]) * x1;
        c0 += bf16_bits_to_f32(row0[kk + 2]) * x2;
        c1 += bf16_bits_to_f32(row1[kk + 2]) * x2;
        d0 += bf16_bits_to_f32(row0[kk + 3]) * x3;
        d1 += bf16_bits_to_f32(row1[kk + 3]) * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += bf16_bits_to_f32(row0[kk]) * xv;
        sum1 += bf16_bits_to_f32(row1[kk]) * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[inline]
fn matvec_rowmajor_block_parallel_bf16(x: &[f32], w_rowmajor: &[bf16], n_rows: usize, k_dim: usize, out: &mut [f32]) {
    let w_rowmajor = bf16_slice_as_u16(w_rowmajor);
    out.par_chunks_mut(MATVEC_BLOCK_ROWS)
        .enumerate()
        .for_each(|(block_idx, out_chunk)| {
            let row_start = block_idx * MATVEC_BLOCK_ROWS;
            let rows = out_chunk.len();
            let w_block = &w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim];
            let mut acc = [0.0f32; MATVEC_BLOCK_ROWS];

            let mut kk = 0usize;
            while kk + 8 <= k_dim {
                let x0 = x[kk];
                let x1 = x[kk + 1];
                let x2 = x[kk + 2];
                let x3 = x[kk + 3];
                let x4 = x[kk + 4];
                let x5 = x[kk + 5];
                let x6 = x[kk + 6];
                let x7 = x[kk + 7];
                for r in 0..rows {
                    let base = r * k_dim + kk;
                    acc[r] += bf16_bits_to_f32(w_block[base]) * x0
                        + bf16_bits_to_f32(w_block[base + 1]) * x1
                        + bf16_bits_to_f32(w_block[base + 2]) * x2
                        + bf16_bits_to_f32(w_block[base + 3]) * x3
                        + bf16_bits_to_f32(w_block[base + 4]) * x4
                        + bf16_bits_to_f32(w_block[base + 5]) * x5
                        + bf16_bits_to_f32(w_block[base + 6]) * x6
                        + bf16_bits_to_f32(w_block[base + 7]) * x7;
                }
                kk += 8;
            }

            while kk < k_dim {
                let xv = x[kk];
                for r in 0..rows {
                    acc[r] += bf16_bits_to_f32(w_block[r * k_dim + kk]) * xv;
                }
                kk += 1;
            }

            out_chunk.copy_from_slice(&acc[..rows]);
        });
}

#[inline]
pub fn matvec_rowmajor_parallel_bf16(x: &[f32], w_rowmajor: &[bf16], n_rows: usize, k_dim: usize, out: &mut [f32]) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            out[i] = dot_unrolled_bf16(x, row);
        }
    } else if n_rows >= MATVEC_BLOCK_THRESHOLD {
        matvec_rowmajor_block_parallel_bf16(x, w_rowmajor, n_rows, k_dim, out);
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            *out_val = dot_unrolled_bf16(x, row);
        });
    }
}

#[inline]
pub(crate) fn dual_matvec_rowmajor_parallel_bf16(
    x: &[f32],
    w0_rowmajor: &[bf16],
    w1_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out0: &mut [f32],
    out1: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w0_rowmajor.len(), n_rows * k_dim, "weight0 size mismatch");
    assert_eq!(w1_rowmajor.len(), n_rows * k_dim, "weight1 size mismatch");
    assert_eq!(out0.len(), n_rows, "out0 size mismatch");
    assert_eq!(out1.len(), n_rows, "out1 size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
            let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (s0, s1) = dot2_unrolled_bf16(x, row0, row1);
            out0[i] = s0;
            out1[i] = s1;
        }
    } else {
        out0.par_iter_mut()
            .zip(out1.par_iter_mut())
            .enumerate()
            .for_each(|(i, (dst0, dst1))| {
                let row0 = &w0_rowmajor[i * k_dim..(i + 1) * k_dim];
                let row1 = &w1_rowmajor[i * k_dim..(i + 1) * k_dim];
                let (s0, s1) = dot2_unrolled_bf16(x, row0, row1);
                *dst0 = s0;
                *dst1 = s1;
            });
    }
}

#[inline]
pub fn dual_matvec_silu_mul_rowmajor_parallel_bf16(
    x: &[f32],
    gate_w_rowmajor: &[bf16],
    up_w_rowmajor: &[bf16],
    n_rows: usize,
    k_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(gate_w_rowmajor.len(), n_rows * k_dim, "gate weight size mismatch");
    assert_eq!(up_w_rowmajor.len(), n_rows * k_dim, "up weight size mismatch");
    assert_eq!(out.len(), n_rows, "out size mismatch");

    if n_rows < MATVEC_PAR_THRESHOLD {
        for i in 0..n_rows {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_bf16(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            out[i] = (g * sig) * u;
        }
    } else {
        out.par_iter_mut().enumerate().for_each(|(i, out_val)| {
            let gate_row = &gate_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let up_row = &up_w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let (g, u) = dot2_unrolled_bf16(x, gate_row, up_row);
            let sig = 1.0 / (1.0 + (-g).exp());
            *out_val = (g * sig) * u;
        });
    }
}

#[inline]
pub fn matvec_argmax_rowmajor_parallel_bf16(x: &[f32], w_rowmajor: &[bf16], n_rows: usize, k_dim: usize) -> usize {
    assert_eq!(x.len(), k_dim, "x len / k_dim mismatch");
    assert_eq!(w_rowmajor.len(), n_rows * k_dim, "weight size mismatch");

    if n_rows >= MATVEC_BLOCK_THRESHOLD {
        let n_blocks = (n_rows + ARGMAX_BLOCK_ROWS - 1) / ARGMAX_BLOCK_ROWS;
        return (0..n_blocks)
            .into_par_iter()
            .map(|block_idx| {
                let row_start = block_idx * ARGMAX_BLOCK_ROWS;
                let rows = (n_rows - row_start).min(ARGMAX_BLOCK_ROWS);
                let w_block = bf16_slice_as_u16(&w_rowmajor[row_start * k_dim..(row_start + rows) * k_dim]);
                let mut acc = [0.0f32; ARGMAX_BLOCK_ROWS];

                let mut kk = 0usize;
                while kk + 8 <= k_dim {
                    let x0 = x[kk];
                    let x1 = x[kk + 1];
                    let x2 = x[kk + 2];
                    let x3 = x[kk + 3];
                    let x4 = x[kk + 4];
                    let x5 = x[kk + 5];
                    let x6 = x[kk + 6];
                    let x7 = x[kk + 7];
                    for r in 0..rows {
                        let base = r * k_dim + kk;
                        acc[r] += bf16_bits_to_f32(w_block[base]) * x0
                            + bf16_bits_to_f32(w_block[base + 1]) * x1
                            + bf16_bits_to_f32(w_block[base + 2]) * x2
                            + bf16_bits_to_f32(w_block[base + 3]) * x3
                            + bf16_bits_to_f32(w_block[base + 4]) * x4
                            + bf16_bits_to_f32(w_block[base + 5]) * x5
                            + bf16_bits_to_f32(w_block[base + 6]) * x6
                            + bf16_bits_to_f32(w_block[base + 7]) * x7;
                    }
                    kk += 8;
                }

                while kk < k_dim {
                    let xv = x[kk];
                    for r in 0..rows {
                        acc[r] += bf16_bits_to_f32(w_block[r * k_dim + kk]) * xv;
                    }
                    kk += 1;
                }

                let mut best = (row_start, f32::NEG_INFINITY);
                for r in 0..rows {
                    let cand = (row_start + r, acc[r]);
                    if cand.1 > best.1 {
                        best = cand;
                    }
                }
                best
            })
            .reduce(|| (0usize, f32::NEG_INFINITY), |a, b| if a.1 >= b.1 { a } else { b })
            .0;
    }

    if n_rows < MATVEC_PAR_THRESHOLD {
        let mut best = (0usize, f32::NEG_INFINITY);
        for i in 0..n_rows {
            let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
            let score = dot_unrolled_bf16(x, row);
            if score > best.1 {
                best = (i, score);
            }
        }
        best.0
    } else {
        (0..n_rows)
            .into_par_iter()
            .map(|i| {
                let row = &w_rowmajor[i * k_dim..(i + 1) * k_dim];
                (i, dot_unrolled_bf16(x, row))
            })
            .reduce(|| (0usize, f32::NEG_INFINITY), |a, b| if a.1 >= b.1 { a } else { b })
            .0
    }
}

// A[..., K] @ B^T, where B is [N(out), K(in)]
// output: [..., N]
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let build_graph = !is_no_grad() && (a.requires_grad() || b.requires_grad());

    let (a_shape, b_shape, a_len) = {
        let ad = a.0.borrow();
        let bd = b.0.borrow();
        (
            ad.data.shape().to_vec(),
            bd.data.shape().to_vec(),
            ad.data.len(),
        )
    };

    if b_shape.len() != 2 {
        panic!("MatMul RHS must be 2D, got {:?}", b_shape);
    }

    let k_dim_a = a_shape[a_shape.len() - 1];
    let n_dim = b_shape[0];
    let k_dim_b = b_shape[1];

    if k_dim_a != k_dim_b {
        panic!(
            "MatMul shape mismatch: a {:?} (K={}) vs b {:?} (K={})",
            a_shape, k_dim_a, b_shape, k_dim_b
        );
    }

    let m_dim = a_len / k_dim_a;

    let decode_no_grad = if m_dim == 1 && !build_graph {
        let ad = a.0.borrow();
        let bd = b.0.borrow();

        let a_owned;
        let a_vec: &[f32] = if let Some(s) = ad.data.as_slice() {
            s
        } else {
            a_owned = ad.data.iter().copied().collect::<Vec<f32>>();
            a_owned.as_slice()
        };

        let b_2d = bd.data.view().into_dimensionality::<Ix2>().unwrap();
        let mut out_shape = a_shape.clone();
        let last_idx = out_shape.len() - 1;
        out_shape[last_idx] = n_dim;

        let mut result = ndarray::ArrayD::<f32>::zeros(IxDyn(&out_shape)).into_shared();
        let out_slice = result
            .as_slice_mut()
            .expect("decode matmul output should be contiguous");

        let b_owned;
        let b_slice: &[f32] = if let Some(s) = b_2d.as_slice() {
            s
        } else {
            b_owned = b_2d.as_standard_layout().to_owned();
            b_owned
                .as_slice()
                .expect("standard-layout matmul RHS should be contiguous")
        };
        matvec_rowmajor_parallel(a_vec, b_slice, n_dim, k_dim_a, out_slice);

        Some(result)
    } else {
        None
    };

    if let Some(result) = decode_no_grad {
        return Tensor::from_data_no_grad(result);
    }

    let res_2d = if m_dim == 1 {
        let ad = a.0.borrow();
        let bd = b.0.borrow();

        let a_owned;
        let a_vec: &[f32] = if let Some(s) = ad.data.as_slice() {
            s
        } else {
            a_owned = ad.data.iter().copied().collect::<Vec<f32>>();
            a_owned.as_slice()
        };

        let b_2d = bd.data.view().into_dimensionality::<Ix2>().unwrap();
        let mut out_vec = vec![0.0f32; n_dim];

        let b_owned;
        let b_slice: &[f32] = if let Some(s) = b_2d.as_slice() {
            s
        } else {
            b_owned = b_2d.as_standard_layout().to_owned();
            b_owned
                .as_slice()
                .expect("standard-layout matmul RHS should be contiguous")
        };
        matvec_rowmajor_parallel(a_vec, b_slice, n_dim, k_dim_a, &mut out_vec);

        Array2::from_shape_vec((1, n_dim), out_vec).expect("decode matvec shape build failed")
    } else {
        let ad = a.0.borrow();
        let bd = b.0.borrow();

        let b_2d = bd.data.view().into_dimensionality::<Ix2>().unwrap();
        let mut res = Array2::<f32>::zeros((m_dim, n_dim));

        if let Ok(a_2d_view) = ad.data.view().into_shape((m_dim, k_dim_a)) {
            general_mat_mul(1.0, &a_2d_view, &b_2d.t(), 0.0, &mut res);
        } else {
            let a_2d_owned = ad
                .data
                .to_owned()
                .into_shape((m_dim, k_dim_a))
                .expect("Reshape A failed");
            general_mat_mul(1.0, &a_2d_owned, &b_2d.t(), 0.0, &mut res);
        }

        res
    };

    let mut out_shape = a_shape.clone();
    let last_idx = out_shape.len() - 1;
    out_shape[last_idx] = n_dim;

    let result = res_2d.into_shape(out_shape).unwrap().into_dyn();

    if !build_graph {
        return Tensor::from_array_no_grad(result);
    }

    let a_clone = a.clone();
    let b_clone = b.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: result.into_shared(),
        grad: None,
        parents: vec![a_clone.clone(), b_clone.clone()],
        requires_grad: true,
        backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let g_len = grad.len();
            let g_m = g_len / n_dim;

            let grad_2d = grad
                .view()
                .into_shape((g_m, n_dim))
                .expect("Grad reshape failed: non-contiguous gradient?");

            let (a_data, b_data) = {
                let ad = a_clone.0.borrow();
                let bd = b_clone.0.borrow();
                (ad.data.clone(), bd.data.clone())
            };

            let a_2d_view = a_data.view().into_shape((m_dim, k_dim_a));
            let a_2d_owned;
            let a_2d = match a_2d_view {
                Ok(v) => v,
                Err(_) => {
                    a_2d_owned = a_data.to_owned().into_shape((m_dim, k_dim_a)).unwrap();
                    a_2d_owned.view()
                }
            };

            let b_2d = b_data.view().into_dimensionality::<Ix2>().unwrap();

            let mut da_2d = Array2::<f32>::zeros((m_dim, k_dim_a));
            general_mat_mul(1.0, &grad_2d, &b_2d, 0.0, &mut da_2d);
            a_clone.add_grad(da_2d.into_shape(a_data.shape()).unwrap().into_dyn());

            let mut db_2d = Array2::<f32>::zeros((n_dim, k_dim_a));
            general_mat_mul(1.0, &grad_2d.t(), &a_2d, 0.0, &mut db_2d);
            b_clone.add_grad(db_2d.into_dyn());
        })),
    })))
}

// lhs: [B, H, M, K]
// rhs: [B, H, K, N]
// out: [B, H, M, N]
pub fn batch_matmul(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let build_graph = !is_no_grad() && (lhs.requires_grad() || rhs.requires_grad());

    let lhs_ref = lhs.data_ref();
    let rhs_ref = rhs.data_ref();

    let lhs_view = lhs_ref.view().into_dimensionality::<Ix4>().unwrap();
    let rhs_view = rhs_ref.view().into_dimensionality::<Ix4>().unwrap();

    let (b, h, m, k) = lhs_view.dim();
    let (b2, h2, k2, n) = rhs_view.dim();

    assert_eq!(b, b2, "batch dim mismatch");
    assert_eq!(h, h2, "head dim mismatch");
    assert_eq!(k, k2, "k dim mismatch");

    let mut output = Array4::<f32>::zeros((b, h, m, n));

    Zip::from(output.outer_iter_mut())
        .and(lhs_view.outer_iter())
        .and(rhs_view.outer_iter())
        .for_each(|mut out_batch, lhs_batch, rhs_batch| {
            Zip::from(out_batch.outer_iter_mut())
                .and(lhs_batch.outer_iter())
                .and(rhs_batch.outer_iter())
                .for_each(|mut out_mat, lhs_mat, rhs_mat| {
                    general_mat_mul(1.0, &lhs_mat, &rhs_mat, 0.0, &mut out_mat);
                });
        });

    let output_dyn = output.into_dyn();

    if !build_graph {
        return Tensor::from_array_no_grad(output_dyn);
    }

    let lhs_clone = lhs.clone();
    let rhs_clone = rhs.clone();

    Tensor(Rc::new(RefCell::new(TensorData {
        data: output_dyn.into_shared(),
        grad: None,
        parents: vec![lhs_clone.clone(), rhs_clone.clone()],
        backward_op: Some(std::rc::Rc::new(move |grad: &ndarray::ArrayViewD<f32>| {
            let grad_view = grad.view().into_dimensionality::<Ix4>().unwrap();
            let l_data = lhs_clone.0.borrow().data.clone();
            let r_data = rhs_clone.0.borrow().data.clone();

            let l_view_4d = l_data.view().into_dimensionality::<Ix4>().unwrap();
            let r_view_4d = r_data.view().into_dimensionality::<Ix4>().unwrap();

            let mut d_lhs = Array4::<f32>::zeros((b, h, m, k));
            Zip::from(d_lhs.outer_iter_mut())
                .and(grad_view.outer_iter())
                .and(r_view_4d.outer_iter())
                .for_each(|mut d_l_b, g_b, r_b| {
                    Zip::from(d_l_b.outer_iter_mut())
                        .and(g_b.outer_iter())
                        .and(r_b.outer_iter())
                        .for_each(|mut d_l_mat, g_mat, r_mat| {
                            general_mat_mul(1.0, &g_mat, &r_mat.t(), 0.0, &mut d_l_mat);
                        });
                });
            lhs_clone.add_grad(d_lhs.into_dyn());

            let mut d_rhs = Array4::<f32>::zeros((b, h, k, n));
            Zip::from(d_rhs.outer_iter_mut())
                .and(l_view_4d.outer_iter())
                .and(grad_view.outer_iter())
                .for_each(|mut d_r_b, l_b, g_b| {
                    Zip::from(d_r_b.outer_iter_mut())
                        .and(l_b.outer_iter())
                        .and(g_b.outer_iter())
                        .for_each(|mut d_r_mat, l_mat, g_mat| {
                            general_mat_mul(1.0, &l_mat.t(), &g_mat, 0.0, &mut d_r_mat);
                        });
                });
            rhs_clone.add_grad(d_rhs.into_dyn());
        })),
        requires_grad: true,
    })))
}
