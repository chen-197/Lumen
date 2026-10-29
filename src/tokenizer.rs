use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::Path;
use tokenizers::Tokenizer as HFTokenizer;

pub struct LlamaTokenizer {
    inner: HFTokenizer,
    byte_tokens: HashMap<usize, u8>,
    special_tokens: HashSet<usize>,
    byte_level_decoder: Option<HashMap<char, u8>>,
}

/// Shared tokenizer API; the old name remains available for compatibility.
pub type ModelTokenizer = LlamaTokenizer;

/// Text-only chat formats supported by the CLI. Arbitrary Jinja/tool templates
/// are not interpreted by this formatter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatFormat {
    TinyLlama,
    Qwen3 { enable_thinking: bool },
}
impl ChatFormat {
    pub fn first_turn(self, system: &str, user: &str) -> String {
        match self {
            Self::TinyLlama => {
                format!("<|system|>\n{system}\n</s>\n<|user|>\n{user}\n</s>\n<|assistant|>\n")
            }
            Self::Qwen3 { .. } => format!(
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n{}",
                self.assistant_prefix()
            ),
        }
    }
    pub fn next_turn(self, user: &str) -> String {
        match self {
            Self::TinyLlama => format!("</s>\n<|user|>\n{user}\n</s>\n<|assistant|>\n"),
            Self::Qwen3 { .. } => format!(
                "<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n{}",
                self.assistant_prefix()
            ),
        }
    }
    fn assistant_prefix(self) -> &'static str {
        match self {
            Self::Qwen3 {
                enable_thinking: false,
            } => "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            _ => "<|im_start|>assistant\n",
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/tokenizer/tests.rs"]
mod tests;

/// A displayable prefix; pending IDs remain part of the original model output.
#[derive(Debug, PartialEq, Eq)]
pub struct DecodedTokenPrefix {
    pub text: String,
    pub consumed_tokens: usize,
    pub pending_token_ids: Vec<usize>,
}

impl LlamaTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let tokenizer =
            HFTokenizer::from_file(path).map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        Ok(Self::from_inner(tokenizer))
    }

    fn from_inner(inner: HFTokenizer) -> Self {
        let byte_level_decoder = serde_json::to_value(inner.get_decoder())
            .ok()
            .filter(|decoder| {
                decoder.get("type").and_then(serde_json::Value::as_str) == Some("ByteLevel")
            })
            .map(|_| {
                // GPT-2/ByteLevel's reversible byte-to-Unicode alphabet.
                let mut extra = 256;
                let mut decoder = HashMap::new();
                for byte in 0..=255u8 {
                    let code = if matches!(byte,33..=126|161..=172|174..=255) {
                        u32::from(byte)
                    } else {
                        let code = extra;
                        extra += 1;
                        code
                    };
                    decoder.insert(char::from_u32(code).unwrap(), byte);
                }
                decoder
            });
        let byte_tokens = (0..=255u8)
            .filter_map(|byte| {
                inner
                    .token_to_id(&format!("<0x{byte:02X}>"))
                    .map(|id| (id as usize, byte))
            })
            .collect();
        let special_tokens = inner
            .get_added_tokens_decoder()
            .into_iter()
            .filter_map(|(id, token)| token.special.then_some(id as usize))
            .collect();
        Self {
            inner,
            byte_tokens,
            special_tokens,
            byte_level_decoder,
        }
    }

    /// Decodes the longest token prefix ending at a complete UTF-8 boundary.
    /// Invalid byte sequences and literal replacement characters are retained.
    /// This does not alter IDs, extend the generation budget, or invent continuations.
    /// Supports `<0xHH>` fallback tokens and ByteLevel decoders; other decoders
    /// retain ordinary decode behavior.
    pub fn decode_complete_prefix(
        &self,
        ids: &[usize],
        skip_special_tokens: bool,
    ) -> DecodedTokenPrefix {
        if let Some(decoder) = &self.byte_level_decoder {
            let mut suffix = Vec::new();
            for (index, &id) in ids.iter().enumerate().rev() {
                if skip_special_tokens && self.special_tokens.contains(&id) {
                    continue;
                }
                let Some(token) = u32::try_from(id)
                    .ok()
                    .and_then(|id| self.inner.id_to_token(id))
                else {
                    break;
                };
                let Some(bytes) = token
                    .chars()
                    .map(|ch| decoder.get(&ch).copied())
                    .collect::<Option<Vec<_>>>()
                else {
                    break;
                };
                suffix.push((index, bytes));
            }
            suffix.reverse();
            let bytes = suffix
                .iter()
                .flat_map(|(_, bytes)| bytes.iter().copied())
                .collect::<Vec<_>>();
            let mut consumed_tokens = ids.len();
            if let Err(error) = std::str::from_utf8(&bytes)
                && error.error_len().is_none()
            {
                let mut end = 0;
                consumed_tokens = suffix.first().map_or(ids.len(), |(index, _)| *index);
                let valid_prefix = std::str::from_utf8(&bytes[..error.valid_up_to()])
                    .expect("UTF-8 error valid_up_to must delimit a valid prefix");
                for (index, token_bytes) in &suffix {
                    end += token_bytes.len();
                    if end > error.valid_up_to() {
                        break;
                    }
                    // One BPE token can finish one character and start the next.
                    // A token boundary preceding the incomplete suffix can itself
                    // split a character, so only commit complete UTF-8 prefixes.
                    if valid_prefix.is_char_boundary(end) {
                        consumed_tokens = index + 1;
                    }
                }
            }
            return DecodedTokenPrefix {
                text: self.decode(&ids[..consumed_tokens], skip_special_tokens),
                consumed_tokens,
                pending_token_ids: ids[consumed_tokens..].to_vec(),
            };
        }
        let mut suffix = Vec::new();
        for (index, &id) in ids.iter().enumerate().rev() {
            if skip_special_tokens && self.special_tokens.contains(&id) {
                continue;
            }
            if let Some(&byte) = self.byte_tokens.get(&id) {
                suffix.push((index, byte));
            } else {
                break;
            }
        }
        suffix.reverse();
        let bytes: Vec<u8> = suffix.iter().map(|&(_, byte)| byte).collect();
        let consumed_tokens = match std::str::from_utf8(&bytes) {
            Err(error) if error.error_len().is_none() => suffix[error.valid_up_to()].0,
            _ => ids.len(),
        };
        DecodedTokenPrefix {
            text: self.decode(&ids[..consumed_tokens], skip_special_tokens),
            consumed_tokens,
            pending_token_ids: ids[consumed_tokens..].to_vec(),
        }
    }

    pub fn encode(
        &self,
        text: &str,
        with_special_tokens: bool,
    ) -> Result<Vec<usize>, Box<dyn Error>> {
        let encoding = self
            .inner
            .encode(text, with_special_tokens)
            .map_err(|e| Box::<dyn Error>::from(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as usize).collect())
    }

    pub fn decode(&self, ids: &[usize], skip_special_tokens: bool) -> String {
        let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&ids_u32, skip_special_tokens)
            .unwrap_or_else(|_| "".to_string())
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.inner.token_to_id(token).map(|id| id as usize)
    }

    pub fn bos_id(&self) -> Option<usize> {
        // 常见 BOS
        for t in ["<s>", "<|begin_of_text|>", "<|endoftext|>"] {
            if let Some(id) = self.token_to_id(t) {
                return Some(id);
            }
        }
        None
    }

    pub fn eos_id(&self) -> Option<usize> {
        // 常见 EOS（Llama2/多数："</s>"；Llama3："<|end_of_text|>"；一些 chat："<|eot_id|>"）
        for t in [
            "</s>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|end_of_text|>",
            "<|eot_id|>",
        ] {
            if let Some(id) = self.token_to_id(t) {
                return Some(id);
            }
        }
        None
    }

    pub fn eot_id(&self) -> Option<usize> {
        // 结束一轮对话的 token（Llama3 instruct 常见）
        self.token_to_id("<|eot_id|>")
            .or_else(|| self.token_to_id("<|im_end|>"))
    }

    pub fn pad_id(&self) -> Option<usize> {
        // 常见 PAD
        for t in ["<pad>", "<|pad|>", "<|endoftext|>"] {
            if let Some(id) = self.token_to_id(t) {
                return Some(id);
            }
        }
        None
    }
}
