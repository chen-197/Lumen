use half::{bf16, f16};
use memmap2::MmapOptions;
use safetensors::tensor::{Dtype, SafeTensors, TensorView, serialize_to_file};
use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    quant_dtype: Dtype,
    manual_scale: Option<f32>,
}

#[derive(Debug)]
struct Entry {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn usage(program: &str) {
    eprintln!(
        "Usage:\n  {program} --input PATH --output PATH [options]\n\nOptions:\n  --dtype DTYPE        Quantized storage dtype: i8 (default: i8)\n  --scale FLOAT        Manual quantization scale override\n"
    );
}

fn parse_dtype(raw: &str) -> Result<Dtype, String> {
    match raw.to_ascii_lowercase().as_str() {
        "i8" | "int8" => Ok(Dtype::I8),
        other => Err(format!("暂不支持的量化 dtype: {other}；当前只支持 i8")),
    }
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().collect();
    let program = argv
        .first()
        .cloned()
        .unwrap_or_else(|| "quantize_safetensors".to_string());
    if argv.len() == 1 {
        usage(&program);
        return Err("缺少参数".to_string());
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut quant_dtype = Dtype::I8;
    let mut manual_scale = None;

    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                usage(&program);
                std::process::exit(0);
            }
            "--input" => {
                i += 1;
                input = Some(PathBuf::from(
                    argv.get(i).ok_or("--input 缺少路径")?.as_str(),
                ));
            }
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(
                    argv.get(i).ok_or("--output 缺少路径")?.as_str(),
                ));
            }
            "--dtype" => {
                i += 1;
                quant_dtype = parse_dtype(argv.get(i).ok_or("--dtype 缺少值")?)?;
            }
            "--scale" => {
                i += 1;
                manual_scale = Some(
                    argv.get(i)
                        .ok_or("--scale 缺少数字")?
                        .parse::<f32>()
                        .map_err(|_| "--scale 需要 f32")?,
                );
            }
            other => return Err(format!("未知参数: {other}")),
        }
        i += 1;
    }

    let input = input.ok_or("必须提供 --input")?;
    let output = output.ok_or("必须提供 --output")?;
    if let Some(scale) = manual_scale
        && (!scale.is_finite() || scale <= 0.0)
    {
        return Err(format!("--scale 必须是有限且 > 0 的数，收到 {scale}"));
    }

    Ok(Args {
        input,
        output,
        quant_dtype,
        manual_scale,
    })
}

fn bytes_from_i8(data: &[i8]) -> Vec<u8> {
    data.iter().map(|&v| v as u8).collect()
}

fn bytes_from_f32(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn decode_f32_bytes(data: &[u8], name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if !data.len().is_multiple_of(4) {
        return Err(format!("{} has invalid F32 byte length {}", name, data.len()).into());
    }
    Ok(data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk should be 4 bytes")))
        .collect())
}

fn decode_f16_bytes(data: &[u8], name: &str) -> Result<Vec<f16>, Box<dyn std::error::Error>> {
    if !data.len().is_multiple_of(2) {
        return Err(format!("{} has invalid F16 byte length {}", name, data.len()).into());
    }
    Ok(data
        .chunks_exact(2)
        .map(|chunk| {
            f16::from_bits(u16::from_le_bytes(
                chunk.try_into().expect("f16 chunk should be 2 bytes"),
            ))
        })
        .collect())
}

fn decode_bf16_bytes(data: &[u8], name: &str) -> Result<Vec<bf16>, Box<dyn std::error::Error>> {
    if !data.len().is_multiple_of(2) {
        return Err(format!("{} has invalid BF16 byte length {}", name, data.len()).into());
    }
    Ok(data
        .chunks_exact(2)
        .map(|chunk| {
            bf16::from_bits(u16::from_le_bytes(
                chunk.try_into().expect("bf16 chunk should be 2 bytes"),
            ))
        })
        .collect())
}

fn finite_max_abs(values: impl Iterator<Item = f32>) -> Result<f32, Box<dyn std::error::Error>> {
    let mut max_abs = 0.0f32;
    for value in values {
        if !value.is_finite() {
            return Err(format!("cannot quantize non-finite value to I8: {value}").into());
        }
        max_abs = max_abs.max(value.abs());
    }
    Ok(max_abs)
}

fn quantize_f32_slice_to_i8(
    data: &[f32],
    scale_override: Option<f32>,
) -> Result<(Vec<i8>, f32), Box<dyn std::error::Error>> {
    let max_abs = finite_max_abs(data.iter().copied())?;
    let scale = if let Some(scale) = scale_override {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("quantization scale must be finite and > 0, got {scale}").into());
        }
        scale
    } else {
        if max_abs > 0.0 {
            (max_abs / 127.0).max(f32::MIN_POSITIVE)
        } else {
            1.0
        }
    };
    let inv_scale = 1.0 / scale;
    let quantized = data
        .iter()
        .map(|&v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    Ok((quantized, scale))
}

fn quantize_f16_slice_to_i8(
    data: &[f16],
    scale_override: Option<f32>,
) -> Result<(Vec<i8>, f32), Box<dyn std::error::Error>> {
    let max_abs = finite_max_abs(data.iter().map(|&v| v.to_f32()))?;
    let scale = if let Some(scale) = scale_override {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("quantization scale must be finite and > 0, got {scale}").into());
        }
        scale
    } else {
        if max_abs > 0.0 {
            (max_abs / 127.0).max(f32::MIN_POSITIVE)
        } else {
            1.0
        }
    };
    let inv_scale = 1.0 / scale;
    let quantized = data
        .iter()
        .map(|&v| (v.to_f32() * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    Ok((quantized, scale))
}

fn quantize_bf16_slice_to_i8(
    data: &[bf16],
    scale_override: Option<f32>,
) -> Result<(Vec<i8>, f32), Box<dyn std::error::Error>> {
    let max_abs = finite_max_abs(data.iter().map(|&v| v.to_f32()))?;
    let scale = if let Some(scale) = scale_override {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(format!("quantization scale must be finite and > 0, got {scale}").into());
        }
        scale
    } else {
        if max_abs > 0.0 {
            (max_abs / 127.0).max(f32::MIN_POSITIVE)
        } else {
            1.0
        }
    };
    let inv_scale = 1.0 / scale;
    let quantized = data
        .iter()
        .map(|&v| (v.to_f32() * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    Ok((quantized, scale))
}

fn quantize_tensor_view(
    name: &str,
    view: &TensorView<'_>,
    scale_override: Option<f32>,
) -> Result<Vec<Entry>, Box<dyn std::error::Error>> {
    let shape = view.shape().to_vec();
    let data = view.data();
    let (quantized, scale) = match view.dtype() {
        Dtype::F32 => quantize_f32_slice_to_i8(&decode_f32_bytes(data, name)?, scale_override)?,
        Dtype::F16 => quantize_f16_slice_to_i8(&decode_f16_bytes(data, name)?, scale_override)?,
        Dtype::BF16 => quantize_bf16_slice_to_i8(&decode_bf16_bytes(data, name)?, scale_override)?,
        other => {
            return Err(format!("{} 不是可量化浮点张量: {:?}", name, other).into());
        }
    };

    Ok(vec![
        Entry {
            name: name.to_string(),
            dtype: Dtype::I8,
            shape,
            bytes: bytes_from_i8(&quantized),
        },
        Entry {
            name: format!("{name}.scale"),
            dtype: Dtype::F32,
            shape: vec![1],
            bytes: bytes_from_f32(&[scale]),
        },
    ])
}

fn existing_i8_tensor_names(
    tensors: &SafeTensors<'_>,
    names: &[String],
) -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let available_names = names.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut i8_names = HashSet::new();
    for name in names {
        if tensors.tensor(name)?.dtype() == Dtype::I8 {
            let scale_name = format!("{name}.scale");
            if !available_names.contains(scale_name.as_str()) {
                return Err(
                    format!("existing I8 tensor {name} is missing companion {scale_name}").into(),
                );
            }
            let scale_view = tensors.tensor(&scale_name)?;
            if scale_view.dtype() != Dtype::F32 || scale_view.shape().iter().product::<usize>() != 1
            {
                return Err(format!(
                    "existing I8 tensor {name} requires scalar F32 companion {scale_name}"
                )
                .into());
            }
            i8_names.insert(name.clone());
        }
    }
    Ok(i8_names)
}

fn is_existing_i8_scale_name(name: &str, i8_names: &HashSet<String>) -> bool {
    name.strip_suffix(".scale")
        .is_some_and(|base_name| i8_names.contains(base_name))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().map_err(|e| format!("参数错误: {e}"))?;
    if args.quant_dtype != Dtype::I8 {
        return Err("当前只支持输出 i8 safetensors".into());
    }
    if !Path::new(&args.input).exists() {
        return Err(format!("输入文件不存在: {}", args.input.display()).into());
    }

    let file = File::open(&args.input)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)?;
    let mut names = tensors
        .names()
        .into_iter()
        .map(|name| name.to_string())
        .collect::<Vec<_>>();
    names.sort();
    let existing_i8_names = existing_i8_tensor_names(&tensors, &names)?;

    println!(
        "Quantizing {} tensors from {} -> {}",
        names.len(),
        args.input.display(),
        args.output.display()
    );

    let mut entries = Vec::with_capacity(names.len() * 2);
    for name in names {
        let view = tensors.tensor(&name)?;
        match view.dtype() {
            _ if is_existing_i8_scale_name(&name, &existing_i8_names) => {
                println!("copied: {name} ({:?}, existing I8 scale)", view.dtype());
                entries.push(Entry {
                    name,
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    bytes: view.data().to_owned(),
                });
            }
            Dtype::F32 | Dtype::F16 | Dtype::BF16 => {
                let mut quantized_entries = quantize_tensor_view(&name, &view, args.manual_scale)?;
                println!("quantized: {name} ({:?} -> I8)", view.dtype());
                entries.append(&mut quantized_entries);
            }
            other => {
                println!("copied: {name} ({:?})", other);
                entries.push(Entry {
                    name,
                    dtype: other,
                    shape: view.shape().to_vec(),
                    bytes: view.data().to_owned(),
                });
            }
        }
    }

    let mut views = Vec::with_capacity(entries.len());
    for entry in &entries {
        let view = TensorView::new(entry.dtype, entry.shape.clone(), &entry.bytes)
            .map_err(|e| format!("构造 TensorView 失败 {}: {}", entry.name, e))?;
        views.push((entry.name.clone(), view));
    }
    serialize_to_file(views, &None, &args.output)?;

    println!("done: {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
