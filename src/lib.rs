#![cfg_attr(
    feature = "x86-fp-kernels-nightly",
    feature(f16, stdarch_x86_avx512_f16)
)]

// src/lib.rs

pub mod arch;
pub mod autograd;
pub mod cpu;
pub mod precision;
#[macro_use]
pub mod module;
pub mod init;
pub mod layers;
pub mod loader;
pub mod loss;
pub mod models;
pub mod ops;
pub mod optim;
pub mod tokenizer;

#[cfg(test)]
#[path = "../tests/validation/accuracy_gate.rs"]
mod accuracy_gate;

#[cfg(test)]
#[path = "../tests/validation/autograd/reuse.rs"]
mod autograd_reuse_tests;

#[cfg(test)]
#[path = "../tests/validation/autograd/gradient_flags.rs"]
mod gradient_flags_tests;

#[cfg(test)]
#[path = "../tests/validation/autograd/accumulation.rs"]
mod gradient_accumulation_tests;

#[cfg(test)]
#[path = "../tests/validation/autograd/state.rs"]
mod tensor_state_tests;

#[cfg(test)]
#[path = "../tests/validation/endurance.rs"]
mod endurance_tests;

#[cfg(test)]
#[path = "../tests/validation/generic_nn.rs"]
mod generic_nn_tests;

#[cfg(test)]
#[path = "../tests/validation/generic_prefill.rs"]
mod generic_prefill_tests;
