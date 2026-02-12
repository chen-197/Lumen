pub mod self_attention;
pub mod encoding; // 对应你的 encoding.rs 文件名

pub use self_attention::{SelfAttention, KVCache};
pub use encoding::RotaryEmbedding;