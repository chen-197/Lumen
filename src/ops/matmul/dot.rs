use super::*;

#[inline]
pub(crate) fn dot_unrolled(x: &[f32], row: &[f32]) -> f32 {
    if let Some(sum) = dot_f32_arch(x, row) {
        return sum;
    }

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
pub(crate) fn dot_unrolled_f32_bf16(x: &[f32], row: &[bf16]) -> f32 {
    if let Some(sum) = dot_f32_bf16_arch(x, row) {
        return sum;
    }

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        s0 += row[kk].to_f32() * x[kk] + row[kk + 4].to_f32() * x[kk + 4];
        s1 += row[kk + 1].to_f32() * x[kk + 1] + row[kk + 5].to_f32() * x[kk + 5];
        s2 += row[kk + 2].to_f32() * x[kk + 2] + row[kk + 6].to_f32() * x[kk + 6];
        s3 += row[kk + 3].to_f32() * x[kk + 3] + row[kk + 7].to_f32() * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk];
        s1 += row[kk + 1].to_f32() * x[kk + 1];
        s2 += row[kk + 2].to_f32() * x[kk + 2];
        s3 += row[kk + 3].to_f32() * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[inline]
pub(crate) fn dot_unrolled_f32_f16(x: &[f32], row: &[f16]) -> f32 {
    if let Some(sum) = dot_f32_f16_arch(x, row) {
        return sum;
    }

    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 8 <= k_dim {
        s0 += row[kk].to_f32() * x[kk] + row[kk + 4].to_f32() * x[kk + 4];
        s1 += row[kk + 1].to_f32() * x[kk + 1] + row[kk + 5].to_f32() * x[kk + 5];
        s2 += row[kk + 2].to_f32() * x[kk + 2] + row[kk + 6].to_f32() * x[kk + 6];
        s3 += row[kk + 3].to_f32() * x[kk + 3] + row[kk + 7].to_f32() * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk];
        s1 += row[kk + 1].to_f32() * x[kk + 1];
        s2 += row[kk + 2].to_f32() * x[kk + 2];
        s3 += row[kk + 3].to_f32() * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk];
        kk += 1;
    }
    sum
}

#[inline]
pub(super) fn dot_unrolled_f32_i8_portable(x: &[f32], row: &[i8], scale: f32) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut s4 = 0.0f32;
    let mut s5 = 0.0f32;
    let mut s6 = 0.0f32;
    let mut s7 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        s0 += row[kk] as f32 * x[kk] + row[kk + 8] as f32 * x[kk + 8];
        s1 += row[kk + 1] as f32 * x[kk + 1] + row[kk + 9] as f32 * x[kk + 9];
        s2 += row[kk + 2] as f32 * x[kk + 2] + row[kk + 10] as f32 * x[kk + 10];
        s3 += row[kk + 3] as f32 * x[kk + 3] + row[kk + 11] as f32 * x[kk + 11];
        s4 += row[kk + 4] as f32 * x[kk + 4] + row[kk + 12] as f32 * x[kk + 12];
        s5 += row[kk + 5] as f32 * x[kk + 5] + row[kk + 13] as f32 * x[kk + 13];
        s6 += row[kk + 6] as f32 * x[kk + 6] + row[kk + 14] as f32 * x[kk + 14];
        s7 += row[kk + 7] as f32 * x[kk + 7] + row[kk + 15] as f32 * x[kk + 15];
        kk += 16;
    }

    while kk + 8 <= k_dim {
        s0 += row[kk] as f32 * x[kk] + row[kk + 4] as f32 * x[kk + 4];
        s1 += row[kk + 1] as f32 * x[kk + 1] + row[kk + 5] as f32 * x[kk + 5];
        s2 += row[kk + 2] as f32 * x[kk + 2] + row[kk + 6] as f32 * x[kk + 6];
        s3 += row[kk + 3] as f32 * x[kk + 3] + row[kk + 7] as f32 * x[kk + 7];
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk] as f32 * x[kk];
        s1 += row[kk + 1] as f32 * x[kk + 1];
        s2 += row[kk + 2] as f32 * x[kk + 2];
        s3 += row[kk + 3] as f32 * x[kk + 3];
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    while kk < k_dim {
        sum += row[kk] as f32 * x[kk];
        kk += 1;
    }
    sum * scale
}

#[inline]
pub(super) fn dot_unrolled_i8_i8(x: &[i8], x_scale: f32, row: &[i8], w_scale: f32) -> f32 {
    if let Some(sum) = dot_i8_i8_arch(x, x_scale, row, w_scale) {
        return sum;
    }

    let mut s0 = 0i32;
    let mut s1 = 0i32;
    let mut s2 = 0i32;
    let mut s3 = 0i32;
    let mut s4 = 0i32;
    let mut s5 = 0i32;
    let mut s6 = 0i32;
    let mut s7 = 0i32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        s0 += row[kk] as i32 * x[kk] as i32 + row[kk + 8] as i32 * x[kk + 8] as i32;
        s1 += row[kk + 1] as i32 * x[kk + 1] as i32 + row[kk + 9] as i32 * x[kk + 9] as i32;
        s2 += row[kk + 2] as i32 * x[kk + 2] as i32 + row[kk + 10] as i32 * x[kk + 10] as i32;
        s3 += row[kk + 3] as i32 * x[kk + 3] as i32 + row[kk + 11] as i32 * x[kk + 11] as i32;
        s4 += row[kk + 4] as i32 * x[kk + 4] as i32 + row[kk + 12] as i32 * x[kk + 12] as i32;
        s5 += row[kk + 5] as i32 * x[kk + 5] as i32 + row[kk + 13] as i32 * x[kk + 13] as i32;
        s6 += row[kk + 6] as i32 * x[kk + 6] as i32 + row[kk + 14] as i32 * x[kk + 14] as i32;
        s7 += row[kk + 7] as i32 * x[kk + 7] as i32 + row[kk + 15] as i32 * x[kk + 15] as i32;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        s0 += row[kk] as i32 * x[kk] as i32 + row[kk + 4] as i32 * x[kk + 4] as i32;
        s1 += row[kk + 1] as i32 * x[kk + 1] as i32 + row[kk + 5] as i32 * x[kk + 5] as i32;
        s2 += row[kk + 2] as i32 * x[kk + 2] as i32 + row[kk + 6] as i32 * x[kk + 6] as i32;
        s3 += row[kk + 3] as i32 * x[kk + 3] as i32 + row[kk + 7] as i32 * x[kk + 7] as i32;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        s0 += row[kk] as i32 * x[kk] as i32;
        s1 += row[kk + 1] as i32 * x[kk + 1] as i32;
        s2 += row[kk + 2] as i32 * x[kk + 2] as i32;
        s3 += row[kk + 3] as i32 * x[kk + 3] as i32;
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    while kk < k_dim {
        sum += row[kk] as i32 * x[kk] as i32;
        kk += 1;
    }
    (sum as f32) * x_scale * w_scale
}

#[inline]
pub(super) fn dot_unrolled_bf16_bf16_scalar(x: &[bf16], row: &[bf16]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk].to_f32();
        s1 += row[kk + 1].to_f32() * x[kk + 1].to_f32();
        s2 += row[kk + 2].to_f32() * x[kk + 2].to_f32();
        s3 += row[kk + 3].to_f32() * x[kk + 3].to_f32();
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[inline]
pub(super) fn dot_unrolled_f16_f16_scalar(x: &[f16], row: &[f16]) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 4 <= k_dim {
        s0 += row[kk].to_f32() * x[kk].to_f32();
        s1 += row[kk + 1].to_f32() * x[kk + 1].to_f32();
        s2 += row[kk + 2].to_f32() * x[kk + 2].to_f32();
        s3 += row[kk + 3].to_f32() * x[kk + 3].to_f32();
        kk += 4;
    }

    let mut sum = s0 + s1 + s2 + s3;
    while kk < k_dim {
        sum += row[kk].to_f32() * x[kk].to_f32();
        kk += 1;
    }
    sum
}

#[inline]
pub(super) fn dot2_unrolled_i8_i8(
    x: &[i8],
    x_scale: f32,
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    if let Some(sum) = dot2_i8_i8_arch(x, x_scale, row0, scale0, row1, scale1) {
        return sum;
    }

    let mut a0 = 0i32;
    let mut a1 = 0i32;
    let mut b0 = 0i32;
    let mut b1 = 0i32;
    let mut c0 = 0i32;
    let mut c1 = 0i32;
    let mut d0 = 0i32;
    let mut d1 = 0i32;
    let mut e0 = 0i32;
    let mut e1 = 0i32;
    let mut f0 = 0i32;
    let mut f1 = 0i32;
    let mut g0 = 0i32;
    let mut g1 = 0i32;
    let mut h0 = 0i32;
    let mut h1 = 0i32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;
        let x8 = x[kk + 8] as i32;
        let x9 = x[kk + 9] as i32;
        let x10 = x[kk + 10] as i32;
        let x11 = x[kk + 11] as i32;
        let x12 = x[kk + 12] as i32;
        let x13 = x[kk + 13] as i32;
        let x14 = x[kk + 14] as i32;
        let x15 = x[kk + 15] as i32;

        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        e0 += row0[kk + 4] as i32 * x4;
        e1 += row1[kk + 4] as i32 * x4;
        f0 += row0[kk + 5] as i32 * x5;
        f1 += row1[kk + 5] as i32 * x5;
        g0 += row0[kk + 6] as i32 * x6;
        g1 += row1[kk + 6] as i32 * x6;
        h0 += row0[kk + 7] as i32 * x7;
        h1 += row1[kk + 7] as i32 * x7;
        a0 += row0[kk + 8] as i32 * x8;
        a1 += row1[kk + 8] as i32 * x8;
        b0 += row0[kk + 9] as i32 * x9;
        b1 += row1[kk + 9] as i32 * x9;
        c0 += row0[kk + 10] as i32 * x10;
        c1 += row1[kk + 10] as i32 * x10;
        d0 += row0[kk + 11] as i32 * x11;
        d1 += row1[kk + 11] as i32 * x11;
        e0 += row0[kk + 12] as i32 * x12;
        e1 += row1[kk + 12] as i32 * x12;
        f0 += row0[kk + 13] as i32 * x13;
        f1 += row1[kk + 13] as i32 * x13;
        g0 += row0[kk + 14] as i32 * x14;
        g1 += row1[kk + 14] as i32 * x14;
        h0 += row0[kk + 15] as i32 * x15;
        h1 += row1[kk + 15] as i32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;

        a0 += row0[kk] as i32 * x0 + row0[kk + 4] as i32 * x4;
        a1 += row1[kk] as i32 * x0 + row1[kk + 4] as i32 * x4;
        b0 += row0[kk + 1] as i32 * x1 + row0[kk + 5] as i32 * x5;
        b1 += row1[kk + 1] as i32 * x1 + row1[kk + 5] as i32 * x5;
        c0 += row0[kk + 2] as i32 * x2 + row0[kk + 6] as i32 * x6;
        c1 += row1[kk + 2] as i32 * x2 + row1[kk + 6] as i32 * x6;
        d0 += row0[kk + 3] as i32 * x3 + row0[kk + 7] as i32 * x7;
        d1 += row1[kk + 3] as i32 * x3 + row1[kk + 7] as i32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        kk += 1;
    }
    (
        (sum0 as f32) * x_scale * scale0,
        (sum1 as f32) * x_scale * scale1,
    )
}

#[inline]
pub(super) fn dot3_unrolled_i8_i8(
    x: &[i8],
    x_scale: f32,
    rows: [I8ScaledSlice<'_>; 3],
) -> (f32, f32, f32) {
    let [row0, row1, row2] = rows;
    if let Some(sum) = dot3_i8_i8_arch(
        x,
        x_scale,
        [
            I8ScaledRow {
                values: row0.values,
                scale: row0.scale,
            },
            I8ScaledRow {
                values: row1.values,
                scale: row1.scale,
            },
            I8ScaledRow {
                values: row2.values,
                scale: row2.scale,
            },
        ],
    ) {
        return sum;
    }
    let scale0 = row0.scale;
    let scale1 = row1.scale;
    let scale2 = row2.scale;
    let row0 = row0.values;
    let row1 = row1.values;
    let row2 = row2.values;

    let mut a0 = 0i32;
    let mut a1 = 0i32;
    let mut a2 = 0i32;
    let mut b0 = 0i32;
    let mut b1 = 0i32;
    let mut b2 = 0i32;
    let mut c0 = 0i32;
    let mut c1 = 0i32;
    let mut c2 = 0i32;
    let mut d0 = 0i32;
    let mut d1 = 0i32;
    let mut d2 = 0i32;
    let mut e0 = 0i32;
    let mut e1 = 0i32;
    let mut e2 = 0i32;
    let mut f0 = 0i32;
    let mut f1 = 0i32;
    let mut f2 = 0i32;
    let mut g0 = 0i32;
    let mut g1 = 0i32;
    let mut g2 = 0i32;
    let mut h0 = 0i32;
    let mut h1 = 0i32;
    let mut h2 = 0i32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;
        let x8 = x[kk + 8] as i32;
        let x9 = x[kk + 9] as i32;
        let x10 = x[kk + 10] as i32;
        let x11 = x[kk + 11] as i32;
        let x12 = x[kk + 12] as i32;
        let x13 = x[kk + 13] as i32;
        let x14 = x[kk + 14] as i32;
        let x15 = x[kk + 15] as i32;

        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        a2 += row2[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        b2 += row2[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        c2 += row2[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        d2 += row2[kk + 3] as i32 * x3;
        e0 += row0[kk + 4] as i32 * x4;
        e1 += row1[kk + 4] as i32 * x4;
        e2 += row2[kk + 4] as i32 * x4;
        f0 += row0[kk + 5] as i32 * x5;
        f1 += row1[kk + 5] as i32 * x5;
        f2 += row2[kk + 5] as i32 * x5;
        g0 += row0[kk + 6] as i32 * x6;
        g1 += row1[kk + 6] as i32 * x6;
        g2 += row2[kk + 6] as i32 * x6;
        h0 += row0[kk + 7] as i32 * x7;
        h1 += row1[kk + 7] as i32 * x7;
        h2 += row2[kk + 7] as i32 * x7;
        a0 += row0[kk + 8] as i32 * x8;
        a1 += row1[kk + 8] as i32 * x8;
        a2 += row2[kk + 8] as i32 * x8;
        b0 += row0[kk + 9] as i32 * x9;
        b1 += row1[kk + 9] as i32 * x9;
        b2 += row2[kk + 9] as i32 * x9;
        c0 += row0[kk + 10] as i32 * x10;
        c1 += row1[kk + 10] as i32 * x10;
        c2 += row2[kk + 10] as i32 * x10;
        d0 += row0[kk + 11] as i32 * x11;
        d1 += row1[kk + 11] as i32 * x11;
        d2 += row2[kk + 11] as i32 * x11;
        e0 += row0[kk + 12] as i32 * x12;
        e1 += row1[kk + 12] as i32 * x12;
        e2 += row2[kk + 12] as i32 * x12;
        f0 += row0[kk + 13] as i32 * x13;
        f1 += row1[kk + 13] as i32 * x13;
        f2 += row2[kk + 13] as i32 * x13;
        g0 += row0[kk + 14] as i32 * x14;
        g1 += row1[kk + 14] as i32 * x14;
        g2 += row2[kk + 14] as i32 * x14;
        h0 += row0[kk + 15] as i32 * x15;
        h1 += row1[kk + 15] as i32 * x15;
        h2 += row2[kk + 15] as i32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        let x4 = x[kk + 4] as i32;
        let x5 = x[kk + 5] as i32;
        let x6 = x[kk + 6] as i32;
        let x7 = x[kk + 7] as i32;

        a0 += row0[kk] as i32 * x0 + row0[kk + 4] as i32 * x4;
        a1 += row1[kk] as i32 * x0 + row1[kk + 4] as i32 * x4;
        a2 += row2[kk] as i32 * x0 + row2[kk + 4] as i32 * x4;
        b0 += row0[kk + 1] as i32 * x1 + row0[kk + 5] as i32 * x5;
        b1 += row1[kk + 1] as i32 * x1 + row1[kk + 5] as i32 * x5;
        b2 += row2[kk + 1] as i32 * x1 + row2[kk + 5] as i32 * x5;
        c0 += row0[kk + 2] as i32 * x2 + row0[kk + 6] as i32 * x6;
        c1 += row1[kk + 2] as i32 * x2 + row1[kk + 6] as i32 * x6;
        c2 += row2[kk + 2] as i32 * x2 + row2[kk + 6] as i32 * x6;
        d0 += row0[kk + 3] as i32 * x3 + row0[kk + 7] as i32 * x7;
        d1 += row1[kk + 3] as i32 * x3 + row1[kk + 7] as i32 * x7;
        d2 += row2[kk + 3] as i32 * x3 + row2[kk + 7] as i32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk] as i32;
        let x1 = x[kk + 1] as i32;
        let x2 = x[kk + 2] as i32;
        let x3 = x[kk + 3] as i32;
        a0 += row0[kk] as i32 * x0;
        a1 += row1[kk] as i32 * x0;
        a2 += row2[kk] as i32 * x0;
        b0 += row0[kk + 1] as i32 * x1;
        b1 += row1[kk + 1] as i32 * x1;
        b2 += row2[kk + 1] as i32 * x1;
        c0 += row0[kk + 2] as i32 * x2;
        c1 += row1[kk + 2] as i32 * x2;
        c2 += row2[kk + 2] as i32 * x2;
        d0 += row0[kk + 3] as i32 * x3;
        d1 += row1[kk + 3] as i32 * x3;
        d2 += row2[kk + 3] as i32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk] as i32;
        sum0 += row0[kk] as i32 * xv;
        sum1 += row1[kk] as i32 * xv;
        sum2 += row2[kk] as i32 * xv;
        kk += 1;
    }
    (
        (sum0 as f32) * x_scale * scale0,
        (sum1 as f32) * x_scale * scale1,
        (sum2 as f32) * x_scale * scale2,
    )
}

#[inline]
pub(crate) fn dot_unrolled_f32_i8(x: &[f32], row: &[i8], scale: f32) -> f32 {
    if let Some(sum) = dot_f32_i8_arch(x, row, scale) {
        sum
    } else {
        dot_unrolled_f32_i8_portable(x, row, scale)
    }
}

#[inline]
pub(crate) fn dot2_unrolled(x: &[f32], row0: &[f32], row1: &[f32]) -> (f32, f32) {
    if let Some(sum) = dot2_f32_arch(x, row0, row1) {
        return sum;
    }

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
pub(crate) fn dot2_unrolled_f32_bf16(x: &[f32], row0: &[bf16], row1: &[bf16]) -> (f32, f32) {
    if let Some(sum) = dot2_f32_bf16_arch(x, row0, row1) {
        return sum;
    }

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

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[inline]
pub(crate) fn dot2_unrolled_f32_f16(x: &[f32], row0: &[f16], row1: &[f16]) -> (f32, f32) {
    if let Some(sum) = dot2_f32_f16_arch(x, row0, row1) {
        return sum;
    }

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

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1)
}

#[inline]
pub(crate) fn dot2_unrolled_f32_i8(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
) -> (f32, f32) {
    if let Some(sum) = dot2_f32_i8_arch(x, row0, scale0, row1, scale1) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        e0 += row0[kk + 4] as f32 * x4;
        e1 += row1[kk + 4] as f32 * x4;
        f0 += row0[kk + 5] as f32 * x5;
        f1 += row1[kk + 5] as f32 * x5;
        g0 += row0[kk + 6] as f32 * x6;
        g1 += row1[kk + 6] as f32 * x6;
        h0 += row0[kk + 7] as f32 * x7;
        h1 += row1[kk + 7] as f32 * x7;
        a0 += row0[kk + 8] as f32 * x8;
        a1 += row1[kk + 8] as f32 * x8;
        b0 += row0[kk + 9] as f32 * x9;
        b1 += row1[kk + 9] as f32 * x9;
        c0 += row0[kk + 10] as f32 * x10;
        c1 += row1[kk + 10] as f32 * x10;
        d0 += row0[kk + 11] as f32 * x11;
        d1 += row1[kk + 11] as f32 * x11;
        e0 += row0[kk + 12] as f32 * x12;
        e1 += row1[kk + 12] as f32 * x12;
        f0 += row0[kk + 13] as f32 * x13;
        f1 += row1[kk + 13] as f32 * x13;
        g0 += row0[kk + 14] as f32 * x14;
        g1 += row1[kk + 14] as f32 * x14;
        h0 += row0[kk + 15] as f32 * x15;
        h1 += row1[kk + 15] as f32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk] as f32 * x0 + row0[kk + 4] as f32 * x4;
        a1 += row1[kk] as f32 * x0 + row1[kk + 4] as f32 * x4;
        b0 += row0[kk + 1] as f32 * x1 + row0[kk + 5] as f32 * x5;
        b1 += row1[kk + 1] as f32 * x1 + row1[kk + 5] as f32 * x5;
        c0 += row0[kk + 2] as f32 * x2 + row0[kk + 6] as f32 * x6;
        c1 += row1[kk + 2] as f32 * x2 + row1[kk + 6] as f32 * x6;
        d0 += row0[kk + 3] as f32 * x3 + row0[kk + 7] as f32 * x7;
        d1 += row1[kk + 3] as f32 * x3 + row1[kk + 7] as f32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1)
}

#[inline]
pub(crate) fn dot3_unrolled(
    x: &[f32],
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_arch(x, row0, row1, row2) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut h2 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk] * x0;
        a1 += row1[kk] * x0;
        a2 += row2[kk] * x0;
        b0 += row0[kk + 1] * x1;
        b1 += row1[kk + 1] * x1;
        b2 += row2[kk + 1] * x1;
        c0 += row0[kk + 2] * x2;
        c1 += row1[kk + 2] * x2;
        c2 += row2[kk + 2] * x2;
        d0 += row0[kk + 3] * x3;
        d1 += row1[kk + 3] * x3;
        d2 += row2[kk + 3] * x3;
        e0 += row0[kk + 4] * x4;
        e1 += row1[kk + 4] * x4;
        e2 += row2[kk + 4] * x4;
        f0 += row0[kk + 5] * x5;
        f1 += row1[kk + 5] * x5;
        f2 += row2[kk + 5] * x5;
        g0 += row0[kk + 6] * x6;
        g1 += row1[kk + 6] * x6;
        g2 += row2[kk + 6] * x6;
        h0 += row0[kk + 7] * x7;
        h1 += row1[kk + 7] * x7;
        h2 += row2[kk + 7] * x7;
        a0 += row0[kk + 8] * x8;
        a1 += row1[kk + 8] * x8;
        a2 += row2[kk + 8] * x8;
        b0 += row0[kk + 9] * x9;
        b1 += row1[kk + 9] * x9;
        b2 += row2[kk + 9] * x9;
        c0 += row0[kk + 10] * x10;
        c1 += row1[kk + 10] * x10;
        c2 += row2[kk + 10] * x10;
        d0 += row0[kk + 11] * x11;
        d1 += row1[kk + 11] * x11;
        d2 += row2[kk + 11] * x11;
        e0 += row0[kk + 12] * x12;
        e1 += row1[kk + 12] * x12;
        e2 += row2[kk + 12] * x12;
        f0 += row0[kk + 13] * x13;
        f1 += row1[kk + 13] * x13;
        f2 += row2[kk + 13] * x13;
        g0 += row0[kk + 14] * x14;
        g1 += row1[kk + 14] * x14;
        g2 += row2[kk + 14] * x14;
        h0 += row0[kk + 15] * x15;
        h1 += row1[kk + 15] * x15;
        h2 += row2[kk + 15] * x15;
        kk += 16;
    }

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
        a2 += row2[kk] * x0 + row2[kk + 4] * x4;
        b0 += row0[kk + 1] * x1 + row0[kk + 5] * x5;
        b1 += row1[kk + 1] * x1 + row1[kk + 5] * x5;
        b2 += row2[kk + 1] * x1 + row2[kk + 5] * x5;
        c0 += row0[kk + 2] * x2 + row0[kk + 6] * x6;
        c1 += row1[kk + 2] * x2 + row1[kk + 6] * x6;
        c2 += row2[kk + 2] * x2 + row2[kk + 6] * x6;
        d0 += row0[kk + 3] * x3 + row0[kk + 7] * x7;
        d1 += row1[kk + 3] * x3 + row1[kk + 7] * x7;
        d2 += row2[kk + 3] * x3 + row2[kk + 7] * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] * x0;
        a1 += row1[kk] * x0;
        a2 += row2[kk] * x0;
        b0 += row0[kk + 1] * x1;
        b1 += row1[kk + 1] * x1;
        b2 += row2[kk + 1] * x1;
        c0 += row0[kk + 2] * x2;
        c1 += row1[kk + 2] * x2;
        c2 += row2[kk + 2] * x2;
        d0 += row0[kk + 3] * x3;
        d1 += row1[kk + 3] * x3;
        d2 += row2[kk + 3] * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] * xv;
        sum1 += row1[kk] * xv;
        sum2 += row2[kk] * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[inline]
pub(crate) fn dot3_unrolled_f32_bf16(
    x: &[f32],
    row0: &[bf16],
    row1: &[bf16],
    row2: &[bf16],
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_bf16_arch(x, row0, row1, row2) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut h2 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        a2 += row2[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        b2 += row2[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        c2 += row2[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        d2 += row2[kk + 3].to_f32() * x3;
        e0 += row0[kk + 4].to_f32() * x4;
        e1 += row1[kk + 4].to_f32() * x4;
        e2 += row2[kk + 4].to_f32() * x4;
        f0 += row0[kk + 5].to_f32() * x5;
        f1 += row1[kk + 5].to_f32() * x5;
        f2 += row2[kk + 5].to_f32() * x5;
        g0 += row0[kk + 6].to_f32() * x6;
        g1 += row1[kk + 6].to_f32() * x6;
        g2 += row2[kk + 6].to_f32() * x6;
        h0 += row0[kk + 7].to_f32() * x7;
        h1 += row1[kk + 7].to_f32() * x7;
        h2 += row2[kk + 7].to_f32() * x7;
        a0 += row0[kk + 8].to_f32() * x8;
        a1 += row1[kk + 8].to_f32() * x8;
        a2 += row2[kk + 8].to_f32() * x8;
        b0 += row0[kk + 9].to_f32() * x9;
        b1 += row1[kk + 9].to_f32() * x9;
        b2 += row2[kk + 9].to_f32() * x9;
        c0 += row0[kk + 10].to_f32() * x10;
        c1 += row1[kk + 10].to_f32() * x10;
        c2 += row2[kk + 10].to_f32() * x10;
        d0 += row0[kk + 11].to_f32() * x11;
        d1 += row1[kk + 11].to_f32() * x11;
        d2 += row2[kk + 11].to_f32() * x11;
        e0 += row0[kk + 12].to_f32() * x12;
        e1 += row1[kk + 12].to_f32() * x12;
        e2 += row2[kk + 12].to_f32() * x12;
        f0 += row0[kk + 13].to_f32() * x13;
        f1 += row1[kk + 13].to_f32() * x13;
        f2 += row2[kk + 13].to_f32() * x13;
        g0 += row0[kk + 14].to_f32() * x14;
        g1 += row1[kk + 14].to_f32() * x14;
        g2 += row2[kk + 14].to_f32() * x14;
        h0 += row0[kk + 15].to_f32() * x15;
        h1 += row1[kk + 15].to_f32() * x15;
        h2 += row2[kk + 15].to_f32() * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        a2 += row2[kk].to_f32() * x0 + row2[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        b2 += row2[kk + 1].to_f32() * x1 + row2[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        c2 += row2[kk + 2].to_f32() * x2 + row2[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        d2 += row2[kk + 3].to_f32() * x3 + row2[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        a2 += row2[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        b2 += row2[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        c2 += row2[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        d2 += row2[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[inline]
pub(crate) fn dot3_unrolled_f32_f16(
    x: &[f32],
    row0: &[f16],
    row1: &[f16],
    row2: &[f16],
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_f16_arch(x, row0, row1, row2) {
        return sum;
    }

    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
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

        a0 += row0[kk].to_f32() * x0 + row0[kk + 4].to_f32() * x4;
        a1 += row1[kk].to_f32() * x0 + row1[kk + 4].to_f32() * x4;
        a2 += row2[kk].to_f32() * x0 + row2[kk + 4].to_f32() * x4;
        b0 += row0[kk + 1].to_f32() * x1 + row0[kk + 5].to_f32() * x5;
        b1 += row1[kk + 1].to_f32() * x1 + row1[kk + 5].to_f32() * x5;
        b2 += row2[kk + 1].to_f32() * x1 + row2[kk + 5].to_f32() * x5;
        c0 += row0[kk + 2].to_f32() * x2 + row0[kk + 6].to_f32() * x6;
        c1 += row1[kk + 2].to_f32() * x2 + row1[kk + 6].to_f32() * x6;
        c2 += row2[kk + 2].to_f32() * x2 + row2[kk + 6].to_f32() * x6;
        d0 += row0[kk + 3].to_f32() * x3 + row0[kk + 7].to_f32() * x7;
        d1 += row1[kk + 3].to_f32() * x3 + row1[kk + 7].to_f32() * x7;
        d2 += row2[kk + 3].to_f32() * x3 + row2[kk + 7].to_f32() * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk].to_f32() * x0;
        a1 += row1[kk].to_f32() * x0;
        a2 += row2[kk].to_f32() * x0;
        b0 += row0[kk + 1].to_f32() * x1;
        b1 += row1[kk + 1].to_f32() * x1;
        b2 += row2[kk + 1].to_f32() * x1;
        c0 += row0[kk + 2].to_f32() * x2;
        c1 += row1[kk + 2].to_f32() * x2;
        c2 += row2[kk + 2].to_f32() * x2;
        d0 += row0[kk + 3].to_f32() * x3;
        d1 += row1[kk + 3].to_f32() * x3;
        d2 += row2[kk + 3].to_f32() * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0;
    let mut sum1 = a1 + b1 + c1 + d1;
    let mut sum2 = a2 + b2 + c2 + d2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk].to_f32() * xv;
        sum1 += row1[kk].to_f32() * xv;
        sum2 += row2[kk].to_f32() * xv;
        kk += 1;
    }
    (sum0, sum1, sum2)
}

#[inline]
pub(super) fn dot3_unrolled_f32_i8_portable(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
) -> (f32, f32, f32) {
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut b0 = 0.0f32;
    let mut b1 = 0.0f32;
    let mut b2 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut c2 = 0.0f32;
    let mut d0 = 0.0f32;
    let mut d1 = 0.0f32;
    let mut d2 = 0.0f32;
    let mut e0 = 0.0f32;
    let mut e1 = 0.0f32;
    let mut e2 = 0.0f32;
    let mut f0 = 0.0f32;
    let mut f1 = 0.0f32;
    let mut f2 = 0.0f32;
    let mut g0 = 0.0f32;
    let mut g1 = 0.0f32;
    let mut g2 = 0.0f32;
    let mut h0 = 0.0f32;
    let mut h1 = 0.0f32;
    let mut h2 = 0.0f32;
    let mut kk = 0usize;
    let k_dim = x.len();

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];

        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        a2 += row2[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        b2 += row2[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        c2 += row2[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        d2 += row2[kk + 3] as f32 * x3;
        e0 += row0[kk + 4] as f32 * x4;
        e1 += row1[kk + 4] as f32 * x4;
        e2 += row2[kk + 4] as f32 * x4;
        f0 += row0[kk + 5] as f32 * x5;
        f1 += row1[kk + 5] as f32 * x5;
        f2 += row2[kk + 5] as f32 * x5;
        g0 += row0[kk + 6] as f32 * x6;
        g1 += row1[kk + 6] as f32 * x6;
        g2 += row2[kk + 6] as f32 * x6;
        h0 += row0[kk + 7] as f32 * x7;
        h1 += row1[kk + 7] as f32 * x7;
        h2 += row2[kk + 7] as f32 * x7;
        a0 += row0[kk + 8] as f32 * x8;
        a1 += row1[kk + 8] as f32 * x8;
        a2 += row2[kk + 8] as f32 * x8;
        b0 += row0[kk + 9] as f32 * x9;
        b1 += row1[kk + 9] as f32 * x9;
        b2 += row2[kk + 9] as f32 * x9;
        c0 += row0[kk + 10] as f32 * x10;
        c1 += row1[kk + 10] as f32 * x10;
        c2 += row2[kk + 10] as f32 * x10;
        d0 += row0[kk + 11] as f32 * x11;
        d1 += row1[kk + 11] as f32 * x11;
        d2 += row2[kk + 11] as f32 * x11;
        e0 += row0[kk + 12] as f32 * x12;
        e1 += row1[kk + 12] as f32 * x12;
        e2 += row2[kk + 12] as f32 * x12;
        f0 += row0[kk + 13] as f32 * x13;
        f1 += row1[kk + 13] as f32 * x13;
        f2 += row2[kk + 13] as f32 * x13;
        g0 += row0[kk + 14] as f32 * x14;
        g1 += row1[kk + 14] as f32 * x14;
        g2 += row2[kk + 14] as f32 * x14;
        h0 += row0[kk + 15] as f32 * x15;
        h1 += row1[kk + 15] as f32 * x15;
        h2 += row2[kk + 15] as f32 * x15;
        kk += 16;
    }

    while kk + 8 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];

        a0 += row0[kk] as f32 * x0 + row0[kk + 4] as f32 * x4;
        a1 += row1[kk] as f32 * x0 + row1[kk + 4] as f32 * x4;
        a2 += row2[kk] as f32 * x0 + row2[kk + 4] as f32 * x4;
        b0 += row0[kk + 1] as f32 * x1 + row0[kk + 5] as f32 * x5;
        b1 += row1[kk + 1] as f32 * x1 + row1[kk + 5] as f32 * x5;
        b2 += row2[kk + 1] as f32 * x1 + row2[kk + 5] as f32 * x5;
        c0 += row0[kk + 2] as f32 * x2 + row0[kk + 6] as f32 * x6;
        c1 += row1[kk + 2] as f32 * x2 + row1[kk + 6] as f32 * x6;
        c2 += row2[kk + 2] as f32 * x2 + row2[kk + 6] as f32 * x6;
        d0 += row0[kk + 3] as f32 * x3 + row0[kk + 7] as f32 * x7;
        d1 += row1[kk + 3] as f32 * x3 + row1[kk + 7] as f32 * x7;
        d2 += row2[kk + 3] as f32 * x3 + row2[kk + 7] as f32 * x7;
        kk += 8;
    }

    while kk + 4 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        a0 += row0[kk] as f32 * x0;
        a1 += row1[kk] as f32 * x0;
        a2 += row2[kk] as f32 * x0;
        b0 += row0[kk + 1] as f32 * x1;
        b1 += row1[kk + 1] as f32 * x1;
        b2 += row2[kk + 1] as f32 * x1;
        c0 += row0[kk + 2] as f32 * x2;
        c1 += row1[kk + 2] as f32 * x2;
        c2 += row2[kk + 2] as f32 * x2;
        d0 += row0[kk + 3] as f32 * x3;
        d1 += row1[kk + 3] as f32 * x3;
        d2 += row2[kk + 3] as f32 * x3;
        kk += 4;
    }

    let mut sum0 = a0 + b0 + c0 + d0 + e0 + f0 + g0 + h0;
    let mut sum1 = a1 + b1 + c1 + d1 + e1 + f1 + g1 + h1;
    let mut sum2 = a2 + b2 + c2 + d2 + e2 + f2 + g2 + h2;
    while kk < k_dim {
        let xv = x[kk];
        sum0 += row0[kk] as f32 * xv;
        sum1 += row1[kk] as f32 * xv;
        sum2 += row2[kk] as f32 * xv;
        kk += 1;
    }
    (sum0 * scale0, sum1 * scale1, sum2 * scale2)
}

#[inline]
pub(crate) fn dot3_unrolled_f32_i8(
    x: &[f32],
    row0: &[i8],
    scale0: f32,
    row1: &[i8],
    scale1: f32,
    row2: &[i8],
    scale2: f32,
) -> (f32, f32, f32) {
    if let Some(sum) = dot3_f32_i8_arch(x, row0, scale0, row1, scale1, row2, scale2) {
        sum
    } else {
        dot3_unrolled_f32_i8_portable(x, row0, scale0, row1, scale1, row2, scale2)
    }
}

#[inline]
pub(super) fn dot3_rows_unrolled_f32_i8(
    x: &[f32],
    weights: I8QkvBlock<'_>,
    k_dim: usize,
    rows: usize,
    out: QkvOutMut<'_>,
) {
    let I8QkvBlock { q, k, v } = weights;
    let q_block = q.values;
    let k_block = k.values;
    let v_block = v.values;
    let q_scale = q.scale;
    let k_scale = k.scale;
    let v_scale = v.scale;
    let QkvOutMut {
        q: q_out,
        k: k_out,
        v: v_out,
    } = out;
    debug_assert!(rows <= 4);
    debug_assert_eq!(q_out.len(), rows);
    debug_assert_eq!(k_out.len(), rows);
    debug_assert_eq!(v_out.len(), rows);

    let mut q_acc = [0.0f32; 4];
    let mut k_acc = [0.0f32; 4];
    let mut v_acc = [0.0f32; 4];
    let mut kk = 0usize;

    while kk + 16 <= k_dim {
        let x0 = x[kk];
        let x1 = x[kk + 1];
        let x2 = x[kk + 2];
        let x3 = x[kk + 3];
        let x4 = x[kk + 4];
        let x5 = x[kk + 5];
        let x6 = x[kk + 6];
        let x7 = x[kk + 7];
        let x8 = x[kk + 8];
        let x9 = x[kk + 9];
        let x10 = x[kk + 10];
        let x11 = x[kk + 11];
        let x12 = x[kk + 12];
        let x13 = x[kk + 13];
        let x14 = x[kk + 14];
        let x15 = x[kk + 15];
        for r in 0..rows {
            let base = r * k_dim + kk;
            q_acc[r] += q_block[base] as f32 * x0
                + q_block[base + 1] as f32 * x1
                + q_block[base + 2] as f32 * x2
                + q_block[base + 3] as f32 * x3
                + q_block[base + 4] as f32 * x4
                + q_block[base + 5] as f32 * x5
                + q_block[base + 6] as f32 * x6
                + q_block[base + 7] as f32 * x7
                + q_block[base + 8] as f32 * x8
                + q_block[base + 9] as f32 * x9
                + q_block[base + 10] as f32 * x10
                + q_block[base + 11] as f32 * x11
                + q_block[base + 12] as f32 * x12
                + q_block[base + 13] as f32 * x13
                + q_block[base + 14] as f32 * x14
                + q_block[base + 15] as f32 * x15;
            k_acc[r] += k_block[base] as f32 * x0
                + k_block[base + 1] as f32 * x1
                + k_block[base + 2] as f32 * x2
                + k_block[base + 3] as f32 * x3
                + k_block[base + 4] as f32 * x4
                + k_block[base + 5] as f32 * x5
                + k_block[base + 6] as f32 * x6
                + k_block[base + 7] as f32 * x7
                + k_block[base + 8] as f32 * x8
                + k_block[base + 9] as f32 * x9
                + k_block[base + 10] as f32 * x10
                + k_block[base + 11] as f32 * x11
                + k_block[base + 12] as f32 * x12
                + k_block[base + 13] as f32 * x13
                + k_block[base + 14] as f32 * x14
                + k_block[base + 15] as f32 * x15;
            v_acc[r] += v_block[base] as f32 * x0
                + v_block[base + 1] as f32 * x1
                + v_block[base + 2] as f32 * x2
                + v_block[base + 3] as f32 * x3
                + v_block[base + 4] as f32 * x4
                + v_block[base + 5] as f32 * x5
                + v_block[base + 6] as f32 * x6
                + v_block[base + 7] as f32 * x7
                + v_block[base + 8] as f32 * x8
                + v_block[base + 9] as f32 * x9
                + v_block[base + 10] as f32 * x10
                + v_block[base + 11] as f32 * x11
                + v_block[base + 12] as f32 * x12
                + v_block[base + 13] as f32 * x13
                + v_block[base + 14] as f32 * x14
                + v_block[base + 15] as f32 * x15;
        }
        kk += 16;
    }

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
            q_acc[r] += q_block[base] as f32 * x0
                + q_block[base + 1] as f32 * x1
                + q_block[base + 2] as f32 * x2
                + q_block[base + 3] as f32 * x3
                + q_block[base + 4] as f32 * x4
                + q_block[base + 5] as f32 * x5
                + q_block[base + 6] as f32 * x6
                + q_block[base + 7] as f32 * x7;
            k_acc[r] += k_block[base] as f32 * x0
                + k_block[base + 1] as f32 * x1
                + k_block[base + 2] as f32 * x2
                + k_block[base + 3] as f32 * x3
                + k_block[base + 4] as f32 * x4
                + k_block[base + 5] as f32 * x5
                + k_block[base + 6] as f32 * x6
                + k_block[base + 7] as f32 * x7;
            v_acc[r] += v_block[base] as f32 * x0
                + v_block[base + 1] as f32 * x1
                + v_block[base + 2] as f32 * x2
                + v_block[base + 3] as f32 * x3
                + v_block[base + 4] as f32 * x4
                + v_block[base + 5] as f32 * x5
                + v_block[base + 6] as f32 * x6
                + v_block[base + 7] as f32 * x7;
        }
        kk += 8;
    }

    while kk < k_dim {
        let xv = x[kk];
        for r in 0..rows {
            let base = r * k_dim + kk;
            q_acc[r] += q_block[base] as f32 * xv;
            k_acc[r] += k_block[base] as f32 * xv;
            v_acc[r] += v_block[base] as f32 * xv;
        }
        kk += 1;
    }

    for r in 0..rows {
        q_out[r] = q_acc[r] * q_scale;
        k_out[r] = k_acc[r] * k_scale;
        v_out[r] = v_acc[r] * v_scale;
    }
}

#[inline]
pub(super) fn dot_unrolled_from_slice(x: &[f32], row: SliceRef<'_>) -> f32 {
    match row {
        SliceRef::F32(row) => dot_unrolled(x, row),
        SliceRef::F16(row) => dot_unrolled_f32_f16(x, row),
        SliceRef::BF16(row) => dot_unrolled_f32_bf16(x, row),
        SliceRef::I8(row, scale) => dot_unrolled_f32_i8(x, row, scale),
    }
}
