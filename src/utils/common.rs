#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use concrete_core::*;
use concrete_core::backends::cuda;
use concrete_core::prelude::*;
use concrete_core::commons::crypto::lwe::LweList as ImplLweList;
use ndarray::prelude::*;
use ndarray::*;
use hdf5::*;
use num::traits::AsPrimitive;

use crate::utils::keys::*;
use crate::utils::luts::*;

use std::borrow::Borrow;
use std::error::Error;
use std::result;
use num::*;

fn convert_to_u64(input: impl Into<u64>) -> u64 {
    input.into()
}

pub fn encrypt_lwe<T: Integer + NumCast>(
    input: &T, 
    log_p: i32, 
    log_q: i32, 
    key: &LweSecretKey64,
    config: &Parameters,
    default_engine: &mut DefaultEngine
) -> Result<LweCiphertext64, Box<dyn Error>> 
{
    let result = (*input).to_u64().unwrap() << (log_q - log_p);
    let result = default_engine.create_plaintext_from(&result)?;
    let result = default_engine.encrypt_lwe_ciphertext(key, &result, config.lwe_var)?;
    Ok(result)
}

pub fn encrypt_lwe_array<T: Integer + NumCast, D: ndarray::Dimension>(
    input: &ArrayView<T, D>,
    log_p: i32, 
    log_q: i32, 
    key: &LweSecretKey64,
    config: &Parameters,
    default_engine: &mut DefaultEngine
) -> Result<Array<LweCiphertext64, D>, Box<dyn Error>> 
{
    let result = input.map(|x| { 
        encrypt_lwe(x, log_p, log_q, key, config, default_engine).unwrap()
    });
    Ok(result)
}

pub fn decrypt_lwe<T: Integer + NumCast>(
    input: &LweCiphertext64, 
    log_p: i32, 
    log_q: i32, 
    key: &LweSecretKey64,
    default_engine: &mut DefaultEngine
) -> Result<T, Box<dyn Error>> {
    let round_off = 1u64 << (log_q - log_p - 1);
    let pt = default_engine.decrypt_lwe_ciphertext(key, input)?;
    let raw = default_engine.retrieve_plaintext(&pt)?;
    let res = (raw + round_off) >> (log_q - log_p);
    let res = T::from(res).unwrap();
    Ok(res)
}

pub fn decrypt_lwe_array<T: Integer + NumCast, D: ndarray::Dimension>(
    input: &ArrayView<LweCiphertext64, D>,
    log_p: i32, 
    log_q: i32, 
    key: &LweSecretKey64,
    default_engine: &mut DefaultEngine
) -> Result<Array<T, D>, Box<dyn Error>> 
{
    let result = input.map(|x| { 
        decrypt_lwe::<T>(x, log_p, log_q, key, default_engine).unwrap()
    });
    Ok(result)
}

pub fn count_different_elts<T: PartialEq, D: ndarray::Dimension>(
    a: &ArrayView<T, D>, 
    b: &ArrayView<T, D>) -> usize
{
    let mut count = 0;
    Zip::from(a).and(b).for_each( |a_i, b_i| {
        if *a_i != *b_i {
            count += 1;
        }
    });
    count
}

pub fn matmul_custom_1D(
    ct: &ArrayView1<LweCiphertext64>, 
    pt: &Array2<i8>, 
    config: &Parameters, 
    default_engine: &mut DefaultEngine
) -> Result<Array1<LweCiphertext64>, Box<dyn Error>> 
{
    // Init acc
    let zero_pt = default_engine.create_plaintext_from(&0u64)?;
    let acc_to_clone: LweCiphertext64 = default_engine.trivially_encrypt_lwe_ciphertext(ct[[0]].lwe_dimension().to_lwe_size(), &zero_pt)?;

    // For each column
    let mut output: Vec<LweCiphertext64> = vec![];
    for column in pt.columns().into_iter(){
        let mut acc = acc_to_clone.clone();
        // IMPORTANT: Using N for LweSize since we are using AP type 1 ^^^^^^^^^^^^^^
        // Multiply c_ij * p_jk and accumulate in acc 
        for (ci, pi) in ct.iter().zip(column.iter()) {
            if *pi < 0 {
                default_engine.fuse_sub_lwe_ciphertext(&mut acc, ci)?;
            }
            if *pi > 0 { 
                default_engine.fuse_add_lwe_ciphertext(&mut acc, ci)?;
            }
        };
       output.push(acc);
    }
    Ok(arr1(&output))
}

pub fn decrypt_lwe_ciphertext_vector(
    input: &LweCiphertextVector64, 
    log_p: i32, 
    log_q: i32, 
    keys: &Keys, 
    default_engine: &mut DefaultEngine
) -> Result<Vec<u64>, Box<dyn Error>> 
{   
    // Rounding after decryption
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Decrypt into ptxts
    let result_pts = default_engine.decrypt_lwe_ciphertext_vector(&keys.lwe, input)?;

    // Get raw u64s
    let result_raw = default_engine.retrieve_plaintext_vector(&result_pts)?;

    // Perform rounding
    let result: Vec<u64> = result_raw
        .iter()
        .map(|x| (x + round_off) >> (log_q - log_p))
        .collect();

    Ok(result)
}

pub fn decrypt_lwe_ciphertexts(
    input: &Vec<LweCiphertext64>, 
    log_p: i32, 
    log_q: i32, 
    keys: &Keys, 
    default_engine: &mut DefaultEngine
) -> Result<Vec<u64>, Box<dyn Error>> 
{   
    // Rounding after decryption
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Decrypt into ptxts
    let result: Vec<u64> = input.iter().map(
        |x| { 
            let pt = default_engine.decrypt_lwe_ciphertext(&keys.lwe, x).unwrap();
            let clrt = default_engine.retrieve_plaintext(&pt).unwrap();
            (clrt + round_off) >> (log_q - log_p) 
        }
    ).collect();

    Ok(result)
}

pub fn populate_lwe_vector(
    ct_arr: Array1<LweCiphertext64>, // Note: deconstructs ct_arr after this function
    default_engine: &mut DefaultEngine
) -> Result<LweCiphertextVector64, Box<dyn Error>> 
{
    let lwe_size = ct_arr[[0]].lwe_dimension().to_lwe_size();
    let mut lwes_raw: Vec<u64> = vec![];
    for lwe in ct_arr.iter() {
        let lwe_raw = default_engine.consume_retrieve_lwe_ciphertext(lwe.clone())?;
        lwes_raw.extend(lwe_raw);
    };
    let result = default_engine.create_lwe_ciphertext_vector_from(lwes_raw, lwe_size)?;
    Ok(result)
}

pub fn depopulate_lwe_vector(
    ct_vec: LweCiphertextVector64,
    default_engine: &mut DefaultEngine
) -> Result<Array1<LweCiphertext64>, Box<dyn Error>>
{
    let chunk_size = ct_vec.lwe_dimension().to_lwe_size().0;
    let mut result_vec: Vec<LweCiphertext64> = vec![];
    let ct_vec_raw = default_engine.consume_retrieve_lwe_ciphertext_vector(ct_vec)?;
    for chunk in ct_vec_raw.chunks(chunk_size) {
        let lwe = default_engine.create_lwe_ciphertext_from(chunk.to_vec())?;
        result_vec.push(lwe);
    }
    Ok(Array::from(result_vec))
}

pub fn amortized_ks_pbs(
    output: &mut CudaLweCiphertextVector64, 
    temp: &mut CudaLweCiphertextVector64,
    input: &CudaLweCiphertextVector64, 
    luts: &CudaGlweCiphertextVector64,
    d_keys: &CudaKeys,
    cuda_engine: &mut CudaEngine,
    amortized_cuda_engine: &mut AmortizedCudaEngine
) -> Result<(), Box<dyn Error>>
{
    // Using AP'1 from https://eprint.iacr.org/2022/704 since it has less cost (DP->KS->PBS)
    cuda_engine.discard_keyswitch_lwe_ciphertext_vector(temp, input, &d_keys.ksk_extracted_lwe)?;
    amortized_cuda_engine.discard_bootstrap_lwe_ciphertext_vector(output, temp, luts, &d_keys.bsk)?;
    Ok(())
}

pub fn amortized_pbs(
    output: &mut CudaLweCiphertextVector64, 
    input: &CudaLweCiphertextVector64, 
    luts: &CudaGlweCiphertextVector64,
    d_keys: &CudaKeys,
    cuda_engine: &mut CudaEngine,
    amortized_cuda_engine: &mut AmortizedCudaEngine
) -> Result<(), Box<dyn Error>>
{
    // Using AP'1 from https://eprint.iacr.org/2022/704 since it has less cost (DP->KS->PBS)
    amortized_cuda_engine.discard_bootstrap_lwe_ciphertext_vector(output, input, luts, &d_keys.bsk)?;
    Ok(())
}

pub fn sign_activation(
    input: Array1<LweCiphertext64>,
    d_temp: &mut CudaLweCiphertextVector64, // Can be persistant (per layer or per equal sized array)
    d_output: &mut CudaLweCiphertextVector64, 
    d_sign_luts: &CudaGlweCiphertextVector64, // Can be persistant (per layer or per equal sized array)
    d_keys: &CudaKeys,
    cuda_engine: &mut CudaEngine,
    amortized_cuda_engine: &mut AmortizedCudaEngine,
    default_engine: &mut DefaultEngine,
) -> Result<Array1<LweCiphertext64>, Box<dyn Error>>
{
    // Create and copy input vector (since im not sure how to do it to an existing one)
    let input_vec = populate_lwe_vector(input, default_engine)?;
    let d_input_vec = cuda_engine.convert_lwe_ciphertext_vector(&input_vec)?;
    
    // Perform sign fn
    amortized_ks_pbs(d_output, d_temp, &d_input_vec, d_sign_luts, d_keys, cuda_engine, amortized_cuda_engine)?;
    
    // Repopulate output array
    let h_output = cuda_engine.convert_lwe_ciphertext_vector(d_output)?;
    let result = depopulate_lwe_vector(h_output, default_engine)?;
    
    Ok(result)
}

pub fn prepare_layer(
    log_p: i32,
    log_q: i32,
    units: usize,
    config: &Parameters,
    cuda_engine: &mut CudaEngine, 
    default_engine: &mut DefaultEngine
) -> Result<(CudaLweCiphertextVector64, CudaLweCiphertextVector64, CudaGlweCiphertextVector64), Box<dyn Error>>
{
    // Create placeholder vector
    let placeholder_pt_vector = default_engine.create_plaintext_vector_from(&vec![0u64; units])?; 
    let placeholder_ct_vector_output = default_engine.trivially_encrypt_lwe_ciphertext_vector(LweSize(config.N.0+1), &placeholder_pt_vector)?; 
    let placeholder_ct_vector_temp = default_engine.trivially_encrypt_lwe_ciphertext_vector(config.n.to_lwe_size(), &placeholder_pt_vector)?; 
    // IMPORTANT: We use N for output (after pbs n->N) and n for temp (after keyswitch N->n) since we do AP type 1 ^^^^^^
    
    // Create device placeholders
    let mut d_output = cuda_engine.convert_lwe_ciphertext_vector(&placeholder_ct_vector_output)?;
    let mut d_temp = cuda_engine.convert_lwe_ciphertext_vector(&placeholder_ct_vector_temp)?;
    
    // Create device luts
    let h_luts = sign_lut(log_p, log_q, d_output.lwe_ciphertext_count().0, &config, default_engine)?;
    let mut d_luts = cuda_engine.convert_glwe_ciphertext_vector(&h_luts)?;
    
    Ok((d_output, d_temp, d_luts))
}



// pub fn ndarray_to_ciphertext_vector_64(ct: &Array1<LweCiphertext64>, config: &Parameters) -> Result<LweCiphertextVector64, Box<dyn Error>> {
//     let mut vector = ImplLweList::allocate(
//         0u64,
//         config.n.to_lwe_size(),
//         CiphertextCount(ct.dim()),
//     );
//     // for (mut empty, cipher) in vector.ciphertext_iter_mut().zip(ct.iter()) {
//     //     empty = ;
//     // }
//     Ok(LweCiphertextVector64(vector))
// }