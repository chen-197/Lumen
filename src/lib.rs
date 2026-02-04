// src/lib.rs

pub mod autograd;
#[macro_use] pub mod module;
pub mod optim;
pub mod mnist_loader;
pub mod loss;
pub mod ops;
pub mod init;
pub mod layers;
pub mod tokenizer;
pub mod models;
pub mod loader;
pub mod kv_cache;