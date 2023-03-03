#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]
#![allow(arithmetic_overflow)]

use concrete_core::*;
use concrete_core::backends::cuda;
use concrete_core::commons::numeric::SignedInteger;
use concrete_core::commons::numeric::UnsignedInteger;
use concrete_core::prelude::*;
use concrete_core::commons::crypto::lwe::LweList as ImplLweList;
use ndarray::prelude::*;
use ndarray::*;
use ndarray_stats::QuantileExt;
use hdf5::*;
use num::traits::RefNum;
use time_graph::{instrument, spanned};

use crate::utils::keys::*;
use crate::utils::luts::*;

use std::borrow::Borrow;
use std::error::Error;
use std::ops::AddAssign;
use std::ops::SubAssign;
use std::result;
use num::*;

fn convert_to_u64(input: impl Into<u64>) -> u64 {
    input.into()
}

#[instrument]
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

#[instrument]
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

#[instrument]
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

#[instrument]
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

#[instrument]
pub fn matmul_custom_1D(
    ct: &ArrayView1<LweCiphertext64>, 
    pt: &ArrayView2<i8>, 
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

#[instrument]
pub fn plaintext_matmul_custom_1D<T: Integer+AddAssign+SubAssign+Clone+Copy, S: Integer>(
    ct: &ArrayView1<T>, 
    pt: &ArrayView2<S>, 
) -> Result<Array1<T>, Box<dyn Error>> 
{
    // For each column
    let mut output: Vec<T> = vec![];
    for column in pt.columns().into_iter(){
        let mut acc = T::zero();
        // Multiply c_ij * p_jk and accumulate in acc 
        for (ci, pi) in ct.iter().zip(column.iter()) {
            if *pi < S::zero() {
                acc -= *ci;
            }
            if *pi > S::zero() { 
                acc += *ci;
            }
        };
       output.push(acc);
    }
    Ok(arr1(&output))
}

#[instrument]
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

#[instrument]
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

#[instrument]
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

#[instrument]
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

#[instrument]
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
    spanned!("cuda_ks", { cuda_engine.discard_keyswitch_lwe_ciphertext_vector(temp, input, &d_keys.ksk_extracted_lwe)?; });
    spanned!("cuda_am_pbs", { amortized_cuda_engine.discard_bootstrap_lwe_ciphertext_vector(output, temp, luts, &d_keys.bsk)?; });
    Ok(())
}

#[instrument]
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

#[instrument]
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
    // amortized_pbs(d_output, &d_input_vec, d_sign_luts, d_keys, cuda_engine, amortized_cuda_engine)?;
    amortized_ks_pbs(d_output, d_temp, &d_input_vec, d_sign_luts, d_keys, cuda_engine, amortized_cuda_engine)?;
    
    // Repopulate output array
    let h_output = cuda_engine.convert_lwe_ciphertext_vector(d_output)?;
    let result = depopulate_lwe_vector(h_output, default_engine)?;
    
    Ok(result)
}

#[instrument]
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

#[instrument]
pub fn encrypted_rnn_block(
    encrypted_input: &ArrayView2<LweCiphertext64>,
    pt_input: &ArrayView2<i32>, // For debugging
    kernel: &ArrayView2<i8>,
    recurrent_kernel: &ArrayView2<i8>,
    layer_name: &str,
    log_p: i32,
    log_q: i32,
    d_keys: &CudaKeys,
    h_keys: &Keys, // For debugging
    config: &Parameters,
    cuda_engine: &mut CudaEngine,
    amortized_cuda_engine: &mut AmortizedCudaEngine,
    default_engine: &mut DefaultEngine
// ) -> Result<Array2<LweCiphertext64>, Box<dyn Error>>
) -> Result<(Array2<LweCiphertext64>, Array2<i32>), Box<dyn Error>> // For debugging
{
    let num_units = kernel.dim().1;
    let (mut d_output, mut d_temp, d_luts) = prepare_layer(log_p, log_q, num_units, config, cuda_engine, default_engine)?;
    let mut states: Array2<LweCiphertext64> = Array2::from_shape_vec((0, num_units), vec![])?;
    let mut pt_states: Array2<i32> = Array2::from_shape_vec((0, num_units), vec![])?; // For debugging
    // for (t, ct_t) in encrypted_input.rows().into_iter().enumerate() {
    for (t, (ct_t, pt_t)) in encrypted_input.rows().into_iter().zip(pt_input.rows().into_iter()).enumerate() { // For debugging
        // W_x * x
        let mut output = matmul_custom_1D(&ct_t, kernel, &config, default_engine)?;
        let mut pt_output = plaintext_matmul_custom_1D(&pt_t, kernel)?; // For debugging

        // DEBUG
        check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: kernel matmul", layer_name, t).as_str(), log_p, log_q, h_keys, default_engine)?;
        
        if t != 0 { // First state is zeros but I don't want to initialize zeros
            // W_h * h
            let mut hidden = matmul_custom_1D(&states.row(t-1) , recurrent_kernel, &config, default_engine)?;
            let mut pt_hidden = plaintext_matmul_custom_1D(&pt_states.row(t-1), recurrent_kernel)?; // For debugging

            // DEBUG
            check_pt_ct_difference(&hidden.view(), &pt_hidden.view(), format!("{}: ts {}: recurrent matmul", layer_name, t).as_str(), log_p, log_q, h_keys, default_engine)?;

            // W_x * x + W_h * h
            for (x_i, h_i) in output.iter_mut().zip(hidden.iter()) {
                default_engine.fuse_add_lwe_ciphertext(x_i, h_i)?;
            }
            pt_output = pt_output + pt_hidden; // For debugging

            // DEBUG
            check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: addition", layer_name, t).as_str(), log_p, log_q, h_keys, default_engine)?;
        }

        // sign(W_x * x + W_h * h)
        output = sign_activation(output, &mut d_temp, &mut d_output, &d_luts, &d_keys, cuda_engine, amortized_cuda_engine, default_engine)?;
        pt_output = pt_output.mapv(|x| sgn_zero_is_one(x)); // For debugging

        // DEBUG
        check_pt_ct_difference(&output.view(), &pt_output.view(), format!("{}: ts {}: activation", layer_name, t).as_str(), log_p, log_q, h_keys, default_engine)?;
        
        states.append(Axis(0), output.view().into_shape((1, num_units))?)?;
        pt_states.append(Axis(0), pt_output.view().into_shape((1, num_units))?)?; // For debugging
    };
    // Ok(states)
    Ok((states, pt_states)) // For debugging
}

pub fn sgn_zero_is_one<T: Signed + AddAssign>(x: T) -> T 
{
    let mut s = x.signum(); 
    if s == T::zero() {s += T::one()};
    s
}

#[instrument]
pub fn encrypted_dense_block(
    encrypted_input: &ArrayView2<LweCiphertext64>,
    pt_input: &ArrayView2<i32>, // For debugging
    kernel: &ArrayView2<i8>,
    layer_name: &str,
    compute_activation: bool,
    log_p: i32,
    log_q: i32,
    d_keys: &CudaKeys,
    h_keys: &Keys, // For debugging
    config: &Parameters,
    cuda_engine: &mut CudaEngine,
    amortized_cuda_engine: &mut AmortizedCudaEngine,
    default_engine: &mut DefaultEngine
// ) -> Result<Array2<LweCiphertext64>, Box<dyn Error>>
) -> Result<(Array2<LweCiphertext64>, Array2<i32>), Box<dyn Error>> // For debuggin
{
    let num_units = kernel.dim().1;
    let (mut d_output, mut d_temp, d_luts) = prepare_layer(log_p, log_q, num_units, config, cuda_engine, default_engine)?;
    let mut states: Array2<LweCiphertext64> = Array2::from_shape_vec((0, num_units), vec![])?;
    let mut pt_states: Array2<i32> = Array2::from_shape_vec((0, num_units), vec![])?; // For debugging
    // for (t, ct_t) in encrypted_input.rows().into_iter().enumerate() {
    for (t, (ct_t, pt_t)) in encrypted_input.rows().into_iter().zip(pt_input.rows().into_iter()).enumerate() { // For debugging
        // W_x * x
        let mut output = matmul_custom_1D(&ct_t, kernel, &config, default_engine)?;
        let mut pt_output = plaintext_matmul_custom_1D(&pt_t, kernel)?; // For debugging

        // sign(W_x * x)
        if compute_activation {
            output = sign_activation(output, &mut d_temp, &mut d_output, &d_luts, &d_keys, cuda_engine, amortized_cuda_engine, default_engine)?;
            pt_output = pt_output.mapv(|x| sgn_zero_is_one(x)); // For debugging
        }

        states.append(Axis(0), output.view().into_shape((1, num_units))?)?;
        pt_states.append(Axis(0), pt_output.view().into_shape((1, num_units))?)?; // For debugging
    };
    // Ok(states)
    Ok((states, pt_states)) // For debugging
}

#[instrument]
pub fn plaintext_rnn_block(
    input: &ArrayView2<i32>,
    kernel: &ArrayView2<i32>,
    recurrent_kernel: &ArrayView2<i32>,
) -> Result<Array2<i32>, Box<dyn Error>>
{
    let num_units = kernel.dim().1;
    let mut states: Array2<i32> = Array2::from_shape_vec((0, num_units), vec![])?;
    for (t, ct_t) in input.rows().into_iter().enumerate() {
        // W_x * x
        // let mut output = ct_t.dot(kernel);
        let mut output = plaintext_matmul_custom_1D(&ct_t, kernel)?;

        if t != 0 { // First state is zeros but I don't want to initialize zeros
            // W_h * h
            // let mut hidden = states.row(t-1).dot(recurrent_kernel);
            let mut hidden = plaintext_matmul_custom_1D(&states.row(t-1), recurrent_kernel)?;
            
            // W_x * x + W_h * h
            for (x_i, h_i) in output.iter_mut().zip(hidden.iter()) {
                *x_i += h_i;
            }
        }

        // sign(W_x * x + W_h * h) where sign(0) = 1
        output = output.mapv(
            |x| { 
                let mut s = x.signum(); 
                if s == 0 {s += 1};
                s
            });
        
        states.append(Axis(0), output.view().into_shape((1, num_units))?)?;
    };
    Ok(states)
}

#[instrument]
pub fn plaintext_dense_block(
    input: &ArrayView2<i32>,
    kernel: &ArrayView2<i32>,
    compute_activation: bool
) -> Result<Array2<i32>, Box<dyn Error>>
{
    let num_units = kernel.dim().1;
    let mut states: Array2<i32> = Array2::from_shape_vec((0, num_units), vec![])?;
    for (t, ct_t) in input.rows().into_iter().enumerate() {
        // W_x * x
        // let mut output = ct_t.dot(kernel);
        let mut output = plaintext_matmul_custom_1D(&ct_t, kernel)?;

        if compute_activation {
            // sign(W_x * x) where sign(0) = 1
            output = output.mapv(
                |x| { 
                    let mut s = x.signum(); 
                    if s == 0 {s += 1};
                    s
                });
        }
        
        states.append(Axis(0), output.view().into_shape((1, num_units))?)?;
    };
    Ok(states)
}

pub fn check_pt_ct_difference<T: Integer+NumCast+Clone, D: ndarray::Dimension>(
    ct: &ArrayView<LweCiphertext64, D>,
    pt: &ArrayView<T, D>,
    check_msg: &str,
    log_p: i32, 
    log_q: i32,
    h_keys: &Keys,
    default_engine: &mut DefaultEngine
) -> Result<(), Box<dyn Error>>
{
    let ct_decrypted: Array<u64, D> = decrypt_lwe_array(ct, log_p, log_q, &h_keys.extracted, default_engine)?;
    let ct_decrypted_i32 = ct_decrypted.mapv(|x| iP_to_iT::<i32>(x, log_p));
    let pt_i32 = pt.mapv(|x| x.to_i32().unwrap() );
    // println!("{:?}\n", ct_decrypted);
    // println!("{:?}\n", ct_decrypted_i32);
    // println!("{:?}\n", pt.mapv(|x| x as i8));
    
    let num_dif_ele = count_different_elts(&ct_decrypted_i32.view(), &pt_i32.view());
    println!("[{}]: Number of different elements = {}, percentage {:.2}%", check_msg, num_dif_ele, 100_f32 * num_dif_ele as f32 / pt_i32.len() as f32);
    
    Ok(())
}

#[instrument]
pub fn time_reduction<T>(
    input: ArrayView2<T>
) -> Result<ArrayView2<T>, Box<dyn Error>>
{
    let (dim_0, dim_1) = input.dim();
    let output = input.into_shape((dim_0/2, dim_1*2))?;
    Ok(output)
}

#[instrument]
pub fn flatten_2D<T>(
    input: ArrayView2<T>
) -> Result<ArrayView2<T>, Box<dyn Error>>
{
    let (dim_0, dim_1) = input.dim();
    let output = input.into_shape((1, dim_0*dim_1))?;
    Ok(output)
}

#[instrument]
pub fn softmax<T: Integer + NumCast + Clone>(a: &Array1<T>) -> Result<Array1<f64>, Box<dyn Error>> {
    let a_f64 = a.mapv(|x| x.to_f64().unwrap());
    let exp_a = a_f64.mapv(f64::exp);
    let sum_exp_a = exp_a.sum();
    Ok(exp_a / sum_exp_a)
}

pub fn compute_softmax_then_argmax<T: Integer + NumCast + Clone>(a: &Array1<T>) -> Result<usize, Box<dyn Error>>
{
    let softmax_output = softmax(a)?;
    let argmax = a.argmax()?;
    Ok(argmax)
}

// Converts a value of signed precision p stored in a u64 to any signed integer you want.
pub fn iP_to_iT<T: SignedInteger + NumCast>(value: u64, p: i32) -> T {
    if (value & (1u64 << (p-1))) != 0 {
        // MSB is 1, so value is negative
        return T::from(value).unwrap() - (T::ONE << p as usize);
    } else {
        // MSB is 0, so value is positive 
        return T::from(value).unwrap();
    }
}