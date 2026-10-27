use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::path::Path;
use tokenizers::Tokenizer as HFTokenizer;

pub struct LlamaTokenizer {
    inner: HFTokenizer,
    byte_tokens: HashMap<usize, u8>,
    special_tokens: HashSet<usize>,
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
        }
    }

    /// Decodes through the last complete UTF-8 character in a trailing byte-token
    /// run. Invalid byte sequences and literal replacement characters are retained.
    /// This does not alter IDs, extend the generation budget, or invent continuations.
    /// Tokenizers without `<0xHH>` fallback tokens retain ordinary decode behavior.
    pub fn decode_complete_prefix(
        &self,
        ids: &[usize],
        skip_special_tokens: bool,
    ) -> DecodedTokenPrefix {
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
        for t in ["<s>", "<|begin_of_text|>"] {
            if let Some(id) = self.token_to_id(t) {
                return Some(id);
            }
        }
        None
    }

    pub fn eos_id(&self) -> Option<usize> {
        // 常见 EOS（Llama2/多数："</s>"；Llama3："<|end_of_text|>"；一些 chat："<|eot_id|>"）
        for t in ["</s>", "<|end_of_text|>", "<|eot_id|>"] {
            if let Some(id) = self.token_to_id(t) {
                return Some(id);
            }
        }
        None
    }

    pub fn eot_id(&self) -> Option<usize> {
        // 结束一轮对话的 token（Llama3 instruct 常见）
        self.token_to_id("<|eot_id|>")
    }

    pub fn pad_id(&self) -> Option<usize> {
        // 常见 PAD
        for t in ["<pad>", "<|pad|>"] {
            if let Some(id) = self.token_to_id(t) {
                return Some(id);
            }
        }
        None
    }
}
