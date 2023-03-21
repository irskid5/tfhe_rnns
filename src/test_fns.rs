#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use concrete_core::commons::math::decomposition::SignedDecomposer;
use concrete_core::commons::utils::ZipChecked;
use concrete_core::prelude::*;
use hdf5::H5Type;
use hdf5::{File, Dataset};
use std::collections::HashMap;
use ndarray::*;
use time_graph::*;

use std::time::Instant;

use std::error::Error;

use crate::utils::keys::*;
use crate::utils::luts::*;
use crate::utils::init::*;
use crate::utils::common::*;

#[macro_export]
macro_rules! print_test_banner {
    ($func_name:ident, $($args:expr),*) => {
        println!("\n--------------------------------------------------");
        println!("Beginning {}", stringify!($func_name).to_uppercase());
        println!("--------------------------------------------------\n");
        $func_name($($args),*)?;
        println!("\n--------------------------------------------------");
        println!("Ending {}", stringify!($func_name).to_uppercase());
        println!("--------------------------------------------------\n");
    };
}

pub fn test_mnist_weights_import_hashmap(filename: &str) -> hdf5::Result<()> {
    // Open the HDF5 file and get the datasets for the weight matrices
    let file = hdf5::File::open(filename)?;

    // Create a HashMap to store the weight matrices
    let mut weight_matrices: HashMap<String, Array2<i8>> = HashMap::new();
    let layers = file.groups()?;

    println!("Opening HDF5 file, is_empty = {:?}", file.is_empty());

    let mut datasets: Vec<Dataset> = Vec::new();
    datasets.push(file.dataset("QRNN_0/QRNN_0/QRNN_0/quantized_kernel:0")?);
    datasets.push(file.dataset("QRNN_0/QRNN_0/QRNN_0/quantized_recurrent_kernel:0")?);
    datasets.push(file.dataset("QRNN_1/QRNN_1/QRNN_1/quantized_kernel:0")?);
    datasets.push(file.dataset("QRNN_1/QRNN_1/QRNN_1/quantized_recurrent_kernel:0")?);
    datasets.push(file.dataset("DENSE_0/DENSE_0/quantized_kernel:0")?);
    datasets.push(file.dataset("DENSE_OUT/DENSE_OUT/quantized_kernel:0")?);

    for dataset in datasets {
        let name = dataset.name();
        println!("{:?}", name);
        if name.contains("quantized") {
            let parts: Vec<&str> = name.split("/").collect();
            let last_two = format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]);
            let data: Vec<i8> = dataset.read_raw()?;
            let shape = dataset.shape();
            let array: Array2<i8> = Array::from_shape_vec((shape[0] as usize, shape[1] as usize), data)?;
            weight_matrices.insert(last_two, array);
        }
    }

    // Print the weight matrices in the HashMap
    for (name, matrix) in weight_matrices.iter() {
        println!("{}:\n{:?}", name, matrix);
    }

    Ok(())
}

pub fn populate_depopulate_lwe_ct_vector(config: &Parameters) -> Result<(), Box<dyn Error>> {
    // Create the necessary engines
    // Here we need to create a secret to give to the unix seeder, but we skip the actual secret creation
    const UNSAFE_SECRET: u128 = 1997;
    let (
        mut default_engine, 
        mut serial_engine, 
        mut parallel_engine, 
        mut cuda_engine, 
        mut amortized_cuda_engine
    ) = init_engines(UNSAFE_SECRET)?;

    // Create keys
    // let keys: Keys = create_keys(config, &mut default_engine, &mut parallel_engine)?;
    // save_keys("./keys/keys.bin", "./keys/", &h_keys, &mut serial_engine)?;
    let keys: Keys = load_keys("./keys/keys.bin", &mut serial_engine)?;

    // Establish precision
    let log_q: i32 = 64;
    let log_p: i32 = 6 + 1;
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Create inputs
    let num_inputs = 10;
    let inputs = vec![5u64; num_inputs];
    let inputs: Array1<u64> = Array::from(inputs);
    let inputs_encrypted = encrypt_lwe_array(&inputs.view(), log_p, log_q, &keys.lwe, config, &mut default_engine)?;

    // Populate vector of lwe ciphertexts ----------------------------------------------------------------------------------------------
    let lwe_vector = populate_lwe_vector(inputs_encrypted, &mut default_engine)?;
    // ---------------------------------------------------------------------------------------------------------------------------------
    
    // Test creation of lwe ciphertext vector
    let lwe_vector_decrypted: Vec<u64> = decrypt_lwe_ciphertext_vector(&lwe_vector, log_p, log_q, &keys, &mut default_engine)?;
    println!("Inputs:      {:?}", inputs);
    println!("Populated:   {:?}", lwe_vector_decrypted);

    // Depopulate lwe ciphertext vector ------------------------------------------------------------------------------------------------
    let lwes_array = depopulate_lwe_vector(lwe_vector, &mut default_engine)?;
    // ---------------------------------------------------------------------------------------------------------------------------------

    let lwes_depopulated_decrypted: Array1<u64> = decrypt_lwe_array(&lwes_array.view(), log_p, log_q, &keys.lwe, &mut default_engine)?;
    println!("Depopulated: {:?}", lwes_depopulated_decrypted);

    Ok(())
}


pub fn amortized_cuda_bs_test(
    decrypt: bool,
    num_cts: usize,
    num_measurements: usize,
    precision: i32,
    config: &Parameters,
) -> Result<(), Box<dyn Error>> {

    // Create the necessary engines
    // Here we need to create a secret to give to the unix seeder, but we skip the actual secret creation
    const UNSAFE_SECRET: u128 = 1997;
    let (
        mut default_engine, 
        mut serial_engine, 
        mut parallel_engine, 
        mut cuda_engine, 
        mut amortized_cuda_engine
    ) = init_engines(UNSAFE_SECRET)?;

    // Create keys
    // let h_keys: Keys = create_keys(config, &mut default_engine, &mut parallel_engine)?;
    // save_keys("./keys/keys.bin", "./keys/", &h_keys, &mut serial_engine)?;
    let h_keys: Keys = load_keys("./keys/keys.bin", &mut serial_engine)?;

    // Establish precision
    let log_q: i32 = 64;
    let log_p: i32 = precision;
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Encrypt inputs/outputs
    let inputs_raw = vec![0_i64; num_cts];   
    // let num_cts: usize = 1 << log_p;
    // let half_range = 1 << (log_p - 1);

    // let inputs_raw: Vec<i64> = (-half_range..half_range).collect(); // The whole domain
    // let inputs_raw: Vec<i64> = vec![-2, -2, 2, 0, 0, 2, -2, -2, 0, 0];
    // let num_cts = inputs_raw.len();
    let inputs: Vec<u64> = inputs_raw.iter().map(|x| (*x as u64) << (log_q - log_p)).collect();
    // println!("inputs: {:?}", inputs_raw);

    let h_inp_pts = default_engine.create_plaintext_vector_from(&inputs)?;
    let h_inp_cts =
        default_engine.encrypt_lwe_ciphertext_vector(&h_keys.lwe, &h_inp_pts, config.lwe_var)?;
    let mut h_out_cts = default_engine.zero_encrypt_lwe_ciphertext_vector(
        &h_keys.extracted,
        config.rlwe_var,
        LweCiphertextCount(num_cts),
    )?;

    // Create LUTs
    let h_luts = sign_mult_lut(log_p, log_q, num_cts, &config, &mut default_engine)?;

    // Send data to GPU (MULTIPLE CT)
    // send input to the GPU
    let d_inp_cts = cuda_engine.convert_lwe_ciphertext_vector(&h_inp_cts)?;
    // convert accumulator to GPU
    let d_luts = cuda_engine.convert_glwe_ciphertext_vector(&h_luts)?;
    // convert BSK to GPU (and from Standard to Fourier representations)
    let d_fourier_bsk: CudaFourierLweBootstrapKey64 =
        cuda_engine.convert_lwe_bootstrap_key(&h_keys.bsk)?;
    let mut d_out_cts = cuda_engine.convert_lwe_ciphertext_vector(&h_out_cts)?;

    println!("Created data and sent to GPU");

    println!("Launching GPU amortized bootstrap of {} LWE CTs.", num_cts);

    let mut measurements: Vec<f32> = vec![];
    for i in 0..num_measurements {
        let now = Instant::now();
        amortized_cuda_engine.discard_bootstrap_lwe_ciphertext_vector(
            &mut d_out_cts,
            &d_inp_cts,
            &d_luts,
            &d_fourier_bsk,
        )?;
        measurements.push(now.elapsed().as_nanos() as f32);
    }
    let avg_ns: f32 = measurements.iter().sum::<f32>() / measurements.len() as f32;
    let avg_ms: f32 = avg_ns * 1e-6;
    // measurements = measurements.iter().map(|x| x * 1e-6).collect();
    // println!("Measurement list: {:?}", measurements);

    println!("Avg duration of {} bootstraps (precision of {}-bits, {} number of measurements) = {:.4} ms", num_cts, log_p-1, num_measurements, avg_ms);

    if decrypt {
        h_out_cts = cuda_engine.convert_lwe_ciphertext_vector(&d_out_cts)?;

        let h_result_pts = default_engine.decrypt_lwe_ciphertext_vector(&h_keys.extracted, &h_out_cts)?;
        let h_result_raw = default_engine.retrieve_plaintext_vector(&h_result_pts)?;
        let h_result: Vec<u64> = h_result_raw.iter().map(|x| (x + round_off) >> (log_q - log_p)).collect();
        println!("Result raw: {:?}", &h_result[0..]);

        let correctness_vec: Vec<i64> = inputs_raw.iter().map(|x| sgn_zero_is_one(*x)).collect();
        let type_matching_result: Vec<i64> = h_result.iter().map(|x| iP_to_iT::<i64>(*x, log_p)).collect();
        println!("Result:          {:?}", type_matching_result);
        println!("Correctness Vec: {:?}", correctness_vec);

        // Calculate correctness
        let mut rights = 0;
        for (inp, res) in correctness_vec.iter().zip(type_matching_result.iter()) {
            if inp == res {
                rights += 1;
            }
        }
        let correctness = 100_f32 * rights as f32 / inputs_raw.len() as f32;

        println!("Correctness ({}-bit) = {:.4} %.", log_p-1, correctness);
    }

    Ok(())
}


pub fn agnes_332() -> Result<(), Box<dyn Error>> {
    let (lwe_dim, lwe_dim_output, glwe_dim, poly_size) = (
        LweDimension(732),
        LweDimension(2048),
        GlweDimension(1),
        PolynomialSize(2048),
        );
    let number_of_inputs = 21846;
    let log_degree = f64::log2(poly_size.0 as f64) as i32;
    let val: u64 = ((poly_size.0 as f64 - (10. * f64::sqrt((lwe_dim.0 as f64) / 16.0)))
                    * 2_f64.powi(64 - log_degree - 1)) as u64;
    let input = vec![val; number_of_inputs];
    let noise = Variance(2_f64.powf(-29.));
    let (dec_lc, dec_bl) = (DecompositionLevelCount(3), DecompositionBaseLog(14));
    // An identity function is applied during the bootstrap
    let mut lut = vec![0u64; poly_size.0 * number_of_inputs];
    for i in 0..poly_size.0 {
        let l = (i as f64 * 2_f64.powi(64 - log_degree - 1)) as u64;
        lut[i] = l;
        lut[i + poly_size.0] = l;
        lut[i + 2 * poly_size.0] = l;
    }

    // 1. default engine
    const UNSAFE_SECRET: u128 = 0;
    let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    let mut default_parallel_engine =
        DefaultParallelEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    // create a vector of LWE ciphertexts
    let h_input_key: LweSecretKey64 = default_engine.generate_new_lwe_secret_key(lwe_dim)?;
    let h_input_plaintext_vector: PlaintextVector64 =
        default_engine.create_plaintext_vector_from(&input)?;
    let h_input_ciphertext_vector: LweCiphertextVector64 = default_engine
        .encrypt_lwe_ciphertext_vector(&h_input_key, &h_input_plaintext_vector, noise)?;
    // create a vector of GLWE ciphertexts containing the encryptions of the LUTs
    let h_lut_plaintext_vector = default_engine.create_plaintext_vector_from(&lut)?;
    let h_lut_key: GlweSecretKey64 =
        default_engine.generate_new_glwe_secret_key(glwe_dim, poly_size)?;
    let h_lut_vector = default_engine.encrypt_glwe_ciphertext_vector(
        &h_lut_key,
        &h_lut_plaintext_vector,
        noise,
        )?;
    // create a BSK
    let h_bootstrap_key: LweBootstrapKey64 = default_parallel_engine.generate_new_lwe_bootstrap_key(
        &h_input_key,
        &h_lut_key,
        dec_bl,
        dec_lc,
        noise,
        )?;
    // initialize an output LWE ciphertext vector
    let h_dummy_key: LweSecretKey64 = default_engine.generate_new_lwe_secret_key(lwe_dim_output)?;

    // 2. cuda engine
    let mut cuda_engine = CudaEngine::new(())?;
    println!("Running on {} GPUs.", cuda_engine.get_number_of_gpus().0);
    let mut cuda_amortized_engine = AmortizedCudaEngine::new(())?;
    // convert input to GPU (split over the GPUs)
    let d_input_ciphertext_vector: CudaLweCiphertextVector64 =
        cuda_engine.convert_lwe_ciphertext_vector(&h_input_ciphertext_vector)?;
    // convert accumulators to GPU
    let d_input_lut_vector: CudaGlweCiphertextVector64 =
        cuda_engine.convert_glwe_ciphertext_vector(&h_lut_vector)?;
    // convert BSK to GPU (and from Standard to Fourier representations)
    let d_fourier_bsk: CudaFourierLweBootstrapKey64 =
        cuda_engine.convert_lwe_bootstrap_key(&h_bootstrap_key)?;
    // launch bootstrap on GPU
    let h_zero_output_ciphertext_vector: LweCiphertextVector64 = default_engine
        .zero_encrypt_lwe_ciphertext_vector(&h_dummy_key, noise, LweCiphertextCount(number_of_inputs))?;
    let mut d_output_ciphertext_vector: CudaLweCiphertextVector64 =
        cuda_engine.convert_lwe_ciphertext_vector(&h_zero_output_ciphertext_vector)?;
    println!("Execute PBS");
    spanned!(
        "pbs", 
        cuda_amortized_engine.discard_bootstrap_lwe_ciphertext_vector(
        &mut d_output_ciphertext_vector,
        &d_input_ciphertext_vector,
        &d_input_lut_vector,
        &d_fourier_bsk,
        )?);
    
    Ok(())

}


pub fn fft_bootstrap_woppbs_test() -> Result<(), Box<dyn Error>> {
    // DISCLAIMER: the parameters used here are only for test purpose, and are not secure.
    let polynomial_size = PolynomialSize(1024);
    let glwe_dimension = GlweDimension(1);
    let lwe_dimension = LweDimension(481);

    let var_small = Variance::from_variance(2f64.powf(-80.0));
    let var_big = Variance::from_variance(2f64.powf(-70.0));

    // Create the required engines
    const UNSAFE_SECRET: u128 = 0;
    let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    let mut default_parallel_engine =
        DefaultParallelEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    let mut fft_engine = FftEngine::new(())?;

    // Generate keys for encryption and evaluation
    let glwe_sk: GlweSecretKey64 =
        default_engine.generate_new_glwe_secret_key(glwe_dimension, polynomial_size)?;
    let lwe_small_sk: LweSecretKey64 = default_engine.generate_new_lwe_secret_key(lwe_dimension)?;
    let lwe_big_sk: LweSecretKey64 =
        default_engine.transform_glwe_secret_key_to_lwe_secret_key(glwe_sk.clone())?;

    let bsk_level_count = DecompositionLevelCount(9);
    let bsk_base_log = DecompositionBaseLog(4);

    let std_bsk: LweBootstrapKey64 = default_parallel_engine.generate_new_lwe_bootstrap_key(
        &lwe_small_sk,
        &glwe_sk,
        bsk_base_log,
        bsk_level_count,
        var_small,
    )?;

    let fourier_bsk: FftFourierLweBootstrapKey64 =
        fft_engine.convert_lwe_bootstrap_key(&std_bsk)?;

    let ksk_level_count = DecompositionLevelCount(9);
    let ksk_base_log = DecompositionBaseLog(1);

    let ksk_big_to_small: LweKeyswitchKey64 = default_engine.generate_new_lwe_keyswitch_key(
        &lwe_big_sk,
        &lwe_small_sk,
        ksk_level_count,
        ksk_base_log,
        var_big,
    )?;

    let pfpksk_level_count = DecompositionLevelCount(9);
    let pfpksk_base_log = DecompositionBaseLog(4);

    let cbs_pfpksk: LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64 = default_engine
        .generate_new_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys(
            &lwe_big_sk,
            &glwe_sk,
            pfpksk_base_log,
            pfpksk_level_count,
            var_small,
        )?;

    // We will have a message with 10 bits of information and we will extract all of them
    let message_bits = 10;
    let bits_to_extract = ExtractedBitsCount(message_bits);

    // The value we encrypt is 42, we will extract the bits of this value and apply the
    // circuit bootstrapping followed by the vertical packing on the extracted bits.
    let cleartext = 42;
    let delta_log_msg = DeltaLog(64 - message_bits);

    // We encode the message on the most significant bits
    let encoded_message = default_engine.create_plaintext_from(&(cleartext << delta_log_msg.0))?;
    let lwe_in = default_engine.encrypt_lwe_ciphertext(&lwe_big_sk, &encoded_message, var_big)?;

    // Bit extraction output, use the zero_encrypt engine to allocate a ciphertext vector
    let mut bit_extraction_output = default_engine.zero_encrypt_lwe_ciphertext_vector(
        &lwe_small_sk,
        var_small,
        LweCiphertextCount(bits_to_extract.0),
    )?;

    // Perform the bit extraction.
    let mut now = Instant::now();
    fft_engine.discard_extract_bits_lwe_ciphertext(
        &mut bit_extraction_output,
        &lwe_in,
        &fourier_bsk,
        &ksk_big_to_small,
        bits_to_extract,
        delta_log_msg,
    )?;
    let mut duration = now.elapsed().as_nanos() as f32;
    println!("Time to extract bits ({} bits) = {:.4} ms", bits_to_extract.0, duration*1e-6);

    // Though the delta log here is the same as the message delta log, in the general case they
    // are different, so we create two DeltaLog parameters
    let delta_log_lut = DeltaLog(64 - message_bits);

    // Create a look-up table we want to apply during vertical packing, here we will perform the
    // addition of the constant 1 and we will apply the right encoding and modulus operation.
    // Adapt the LUT generation to your usage.
    // Here we apply a single look-up table as we output a single ciphertext.
    let number_of_luts_and_output_vp_ciphertexts = 1;
    let lut_size = 1 << bits_to_extract.0;
    let mut lut: Vec<u64> = Vec::with_capacity(lut_size);

    for i in 0..lut_size {
        lut.push(((i as u64 + 1) % (1 << message_bits)) << delta_log_lut.0);
    }

    let lut_as_plaintext_vector = default_engine.create_plaintext_vector_from(lut.as_slice())?;

    // We run on views, so we need a container for the output
    let mut output_cbs_vp_ct_container = vec![
        0u64;
        lwe_big_sk.lwe_dimension().to_lwe_size().0
            * number_of_luts_and_output_vp_ciphertexts
    ];

    let mut output_cbs_vp_ct_mut_view: LweCiphertextVectorMutView64 = default_engine
        .create_lwe_ciphertext_vector_from(
            output_cbs_vp_ct_container.as_mut_slice(),
            lwe_big_sk.lwe_dimension().to_lwe_size(),
        )?;

    // And we need to get a view on the bits extracted earlier that serve as inputs to the
    // circuit bootstrap + vertical packing
    let extracted_bits_lwe_size = bit_extraction_output.lwe_dimension().to_lwe_size();
    let extracted_bits_container =
        default_engine.consume_retrieve_lwe_ciphertext_vector(bit_extraction_output)?;
    let cbs_vp_input_vector_view: LweCiphertextVectorView64 = default_engine
        .create_lwe_ciphertext_vector_from(
            extracted_bits_container.as_slice(),
            extracted_bits_lwe_size,
        )?;

    let cbs_level_count = DecompositionLevelCount(4);
    let cbs_base_log = DecompositionBaseLog(6);
    
    now = Instant::now();
    fft_engine.discard_circuit_bootstrap_boolean_vertical_packing_lwe_ciphertext_vector(
        &mut output_cbs_vp_ct_mut_view,
        &cbs_vp_input_vector_view,
        &fourier_bsk,
        &lut_as_plaintext_vector,
        cbs_level_count,
        cbs_base_log,
        &cbs_pfpksk,
    )?;
    duration = now.elapsed().as_nanos() as f32;
    println!("Time for circuit bootstrap = {:.4} ms", duration*1e-6);

    let lwe_ciphertext_vector_container_as_slice =
        &*default_engine.consume_retrieve_lwe_ciphertext_vector(output_cbs_vp_ct_mut_view)?;

    let output_cbs_vp_ct_view: LweCiphertextVectorView64 = default_engine
        .create_lwe_ciphertext_vector_from(
            lwe_ciphertext_vector_container_as_slice,
            lwe_big_sk.lwe_dimension().to_lwe_size(),
        )?;

    let decrypted_output =
        default_engine.decrypt_lwe_ciphertext_vector(&lwe_big_sk, &output_cbs_vp_ct_view)?;
    let decrypted_plaintext = default_engine.retrieve_plaintext_vector(&decrypted_output)?;

    // We want to work on 10 bits values, so pick a decomposer for 1 single level of 10 bits
    let decomposer =
        SignedDecomposer::new(DecompositionBaseLog(10), DecompositionLevelCount(1));

    let rounded_output = decomposer.closest_representable(decrypted_plaintext[0]);

    let decoded_output = rounded_output >> delta_log_lut.0;

    // 42 + 1 == 43 in our 10 bits output ciphertext
    assert_eq!(decoded_output, 43);

    Ok(())
}