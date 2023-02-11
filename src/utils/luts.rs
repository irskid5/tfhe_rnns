#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use concrete_core::prelude::*;

use crate::utils::keys::Parameters;

pub fn sign_lut(
    log_p: i32,
    log_q: i32,
    num_cts: usize,
    config: &Parameters,
    default_engine: &mut DefaultEngine,
) -> GlweCiphertextVector64 {
    let luts = vec![1u64 << (log_q - log_p); num_cts * config.N.0]; // you create a num_ct*glwe_size vector for multiple glwe cts in a vector
    let lut_pts = default_engine.create_plaintext_vector_from(&luts).unwrap();

    default_engine
        .trivially_encrypt_glwe_ciphertext_vector(
            config.k.to_glwe_size(),
            GlweCiphertextCount(num_cts),
            &lut_pts,
        )
        .unwrap()
}
