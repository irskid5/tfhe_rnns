#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use concrete_core::prelude::*;

use std::time::Instant;

use std::error::Error;

use crate::utils::keys::*;
use crate::utils::luts::*;

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

pub fn amortized_cuda_bs_test(
    decrypt: bool,
    num_cts: usize,
    num_measurements: usize,
) -> Result<(), Box<dyn Error>> {
    // Params from
    // let config = Parameters {
    //     n: LweDimension(774),
    //     lwe_var: Variance(StandardDev(0.000002886954936071319246944).get_variance()),
    //     N: PolynomialSize(2048),
    //     k: GlweDimension(1),
    //     rlwe_var: Variance(
    //         StandardDev(0.00000000000000022148688116005568513645324585951).get_variance(),
    //     ),
    //     l_pbs: DecompositionLevelCount(1),
    //     Bg_bit_pbs: DecompositionBaseLog(16),
    //     l_ks: DecompositionLevelCount(5),
    //     base_bit_ks: DecompositionBaseLog(4),
    // };
    let config = Parameters {
        n: LweDimension(774),
        lwe_var: Variance(StandardDev(0.000002886954936071319246944).get_variance()),
        N: PolynomialSize(2048),
        k: GlweDimension(1),
        rlwe_var: Variance(
            StandardDev(0.00000000000000022148688116005568513645324585951).get_variance(),
        ),
        l_pbs: DecompositionLevelCount(1),
        Bg_bit_pbs: DecompositionBaseLog(16),
        l_ks: DecompositionLevelCount(5),
        base_bit_ks: DecompositionBaseLog(4),
    };

    // Create the necessary engines
    // Here we need to create a secret to give to the unix seeder, but we skip the actual secret creation
    const UNSAFE_SECRET: u128 = 1997;
    let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    let mut serial_engine = DefaultSerializationEngine::new(())?;
    let mut parallel_engine = DefaultParallelEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET)))?;
    let mut cuda_engine = CudaEngine::new(())?;
    let mut amortized_cuda_engine = AmortizedCudaEngine::new(())?;
    println!("Constructed Engines.");

    // Create keys
    let h_keys: Keys = create_keys(&config, &mut default_engine, &mut parallel_engine)?;

    // Establish precision
    let log_q: i32 = 64;
    let log_p: i32 = 6;
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Encrypt inputs/outputs
    let inputs: Vec<u64> = vec![10u64 << (log_q - log_p); num_cts];
    let h_inp_pts = default_engine.create_plaintext_vector_from(&inputs)?;
    let h_inp_cts =
        default_engine.encrypt_lwe_ciphertext_vector(&h_keys.lwe, &h_inp_pts, config.lwe_var)?;
    let mut h_out_cts = default_engine.zero_encrypt_lwe_ciphertext_vector(
        &h_keys.extracted,
        config.rlwe_var,
        LweCiphertextCount(num_cts),
    )?;

    // Create LUTs
    let h_luts = sign_lut(log_p, log_q, num_cts, &config, &mut default_engine);

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

    println!(
        "Avg duration of {} bootstraps (precision of {}-bits, {} number of measurements) = {:.4}ms",
        num_cts, log_p, num_measurements, avg_ms
    );

    if decrypt {
        h_out_cts = cuda_engine.convert_lwe_ciphertext_vector(&d_out_cts)?;

        let h_result_pts =
            default_engine.decrypt_lwe_ciphertext_vector(&h_keys.extracted, &h_out_cts)?;

        let h_result_raw = default_engine.retrieve_plaintext_vector(&h_result_pts)?;

        let h_result: Vec<u64> = h_result_raw
            .iter()
            .map(|x| (x + round_off) >> (log_q - log_p))
            .collect();

        println!("Result = {:?}", &h_result[0..]);
    }

    Ok(())
}
