use super::*;
use std::collections::BTreeMap;
use std::path::{Component, PathBuf};

impl ModelLoader {
    /// Load canonical parameter names from one safetensors file or a HF shard index.
    /// Missing tensors, invalid metadata and shapes are rejected before any load.
    pub fn load_weights_with_options(
        path: impl AsRef<Path>,
        parameters: &HashMap<String, Tensor>,
        options: WeightLoadOptions,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let path = if path.is_dir() {
            let single = path.join("model.safetensors");
            if single.is_file() {
                single
            } else {
                path.join("model.safetensors.index.json")
            }
        } else {
            path.to_owned()
        };
        let mut groups: BTreeMap<PathBuf, HashMap<String, Tensor>> = BTreeMap::new();
        if path.extension().is_some_and(|ext| ext == "json") {
            #[derive(Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let index: Index = serde_json::from_reader(File::open(&path)?)?;
            for (name, tensor) in parameters {
                let filename = index
                    .weight_map
                    .get(name)
                    .ok_or_else(|| format!("Missing required weight in shard index: {name}"))?;
                let relative = Path::new(filename);
                if relative.components().count() != 1
                    || !relative
                        .components()
                        .all(|c| matches!(c, Component::Normal(_)))
                {
                    return Err(format!("Invalid shard filename: {filename}").into());
                }
                groups
                    .entry(path.parent().unwrap_or(Path::new(".")).join(relative))
                    .or_default()
                    .insert(name.clone(), tensor.clone());
            }
        } else {
            groups.insert(path, parameters.clone());
        }
        for (file, params) in &groups {
            let handle = File::open(file)?;
            let mmap = unsafe { MmapOptions::new().map(&handle)? };
            let tensors = SafeTensors::deserialize(&mmap)?;
            for (name, target) in params {
                let view = tensors
                    .tensor(name)
                    .map_err(|_| format!("Missing required weight: {name}"))?;
                if view.shape() != target.shape_vec() {
                    return Err(format!(
                        "Shape mismatch for {name}: checkpoint={:?}, target={:?}",
                        view.shape(),
                        target.shape_vec()
                    )
                    .into());
                }
                match view.dtype() {
                    safetensors::Dtype::F32
                    | safetensors::Dtype::F16
                    | safetensors::Dtype::BF16 => (),
                    safetensors::Dtype::I8 => {
                        let scale =
                            Self::find_i8_scale_tensor_name(&tensors, name).ok_or_else(|| {
                                format!("I8 tensor {name} is missing required companion scale")
                            })?;
                        Self::read_scalar_scale(&tensors, &scale)?;
                    }
                    dtype => return Err(format!("Unsupported dtype {dtype:?} for {name}").into()),
                }
            }
        }
        for (file, params) in groups {
            Self::load_llama_weights_with_options(file, &params, options)?;
        }
        Ok(())
    }
}
