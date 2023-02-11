#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use bincode::{deserialize_from, serialize_into};
use concrete_core::prelude::*;

use std::error::Error;
use std::fs;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::mem::*;
use std::time::Instant;

#[derive(Debug)]
pub struct Parameters {
    n: LweDimension,
    lwe_var: Variance,
    N: PolynomialSize,
    k: GlweDimension,
    rlwe_var: Variance,
    l_pbs: DecompositionLevelCount,
    Bg_bit_pbs: DecompositionBaseLog,
    l_ks: DecompositionLevelCount,
    base_bit_ks: DecompositionBaseLog,
}

#[derive(Debug)]
pub struct Keys {
    lwe: LweSecretKey64,
    glwe: GlweSecretKey64,
    extracted: LweSecretKey64,
    ksk_extracted_lwe: LweKeyswitchKey64,
    bsk: LweBootstrapKey64,
}

fn main() -> Result<(), Box<dyn Error>> {
    // Params from
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
    let mut default_engine = DefaultEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET))).unwrap();
    let mut serial_engine = DefaultSerializationEngine::new(()).unwrap();
    let mut parallel_engine =
        DefaultParallelEngine::new(Box::new(UnixSeeder::new(UNSAFE_SECRET))).unwrap();
    let mut cuda_engine = CudaEngine::new(()).unwrap();
    let mut amortized_cuda_engine = AmortizedCudaEngine::new(()).unwrap();
    println!("Constructed Engines.");

    // let h_keys: Keys = create_keys(&config, &mut default_engine, &mut parallel_engine);
    // save_keys("./keys/keys.bin", "./keys", &h_keys, &mut serial_engine);
    let h_keys: Keys = load_keys("./keys/keys.bin", &mut serial_engine);
    print_key_info(&h_keys);

    // Establish precision
    let log_q: i32 = 64;
    let log_p: i32 = 7;
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Encrypt vector of values for whole distribution
    let num_cts = 1 << log_p;
    let half_range = 1 << (log_p - 1);

    let inputs_raw: Vec<i64> = (-half_range..half_range).collect(); // The whole domain
    let inputs: Vec<u64> = inputs_raw
        .iter()
        .map(|x| (*x as u64) << (log_q - log_p))
        .collect();
    // let inputs = vec![((-20i8 as u8) as u64) << (log_q - log_p); num_cts];
    let h_inp_pts = default_engine
        .create_plaintext_vector_from(&inputs)
        .unwrap();
    let h_inp_cts = default_engine
        .encrypt_lwe_ciphertext_vector(&h_keys.lwe, &h_inp_pts, config.lwe_var)
        .unwrap();
    let mut h_out_cts = default_engine
        .zero_encrypt_lwe_ciphertext_vector(
            &h_keys.extracted,
            config.rlwe_var,
            LweCiphertextCount(num_cts),
        )
        .unwrap();

    // Create LUTs
    // let lut = vec![1u64 << (log_q - log_p); N.0];
    let luts = vec![1u64 << (log_q - log_p); num_cts * config.N.0]; // you create a num_ct*glwe_size vector for multiple glwe cts in a vector
    let h_lut_pts = default_engine.create_plaintext_vector_from(&luts).unwrap();
    let h_luts = default_engine
        .trivially_encrypt_glwe_ciphertext_vector(
            config.k.to_glwe_size(),
            GlweCiphertextCount(num_cts),
            &h_lut_pts,
        )
        .unwrap();

    // Send data to GPU (MULTIPLE CT)
    // send input to the GPU
    let d_inp_cts = cuda_engine
        .convert_lwe_ciphertext_vector(&h_inp_cts)
        .unwrap();
    // convert accumulator to GPU
    let d_luts = cuda_engine.convert_glwe_ciphertext_vector(&h_luts).unwrap();
    // convert BSK to GPU (and from Standard to Fourier representations)
    let d_fourier_bsk: CudaFourierLweBootstrapKey64 =
        cuda_engine.convert_lwe_bootstrap_key(&h_keys.bsk).unwrap();
    let mut d_out_cts = cuda_engine
        .convert_lwe_ciphertext_vector(&h_out_cts)
        .unwrap();

    println!("Created data and sent to GPU");

    // launch the bootstrap on GPU (ONE CT)
    // println!("Launching GPU bootstrap 1 LWE op.");
    // let now = Instant::now();
    // cuda_engine
    //     .discard_bootstrap_lwe_ciphertext(
    //         &mut d_output_ciphertext,
    //         &d_input_ciphertext,
    //         &d_input_lut,
    //         &d_fourier_bsk,
    //     )
    //     .unwrap();
    // let duration = now.elapsed().as_micros();
    // println!(
    //     "Duration of 1 bootstrap (precision of {}-bits) = {}",
    //     log_p, duration
    // );

    println!(
        "Launching GPU amortized bootstrap of {} LWE CTs op.",
        num_cts
    );
    let now = Instant::now();
    amortized_cuda_engine
        .discard_bootstrap_lwe_ciphertext_vector(
            &mut d_out_cts,
            &d_inp_cts,
            &d_luts,
            &d_fourier_bsk,
        )
        .unwrap();
    let duration = now.elapsed().as_micros();
    println!(
        "Duration of {} bootstrap (precision of {}-bits) = {}",
        num_cts, log_p, duration
    );

    h_out_cts = cuda_engine
        .convert_lwe_ciphertext_vector(&d_out_cts)
        .unwrap();

    let h_result_pts = default_engine
        .decrypt_lwe_ciphertext_vector(&h_keys.extracted, &h_out_cts)
        .unwrap();

    let h_result_raw = default_engine
        .retrieve_plaintext_vector(&h_result_pts)
        .unwrap();

    let h_result: Vec<u64> = h_result_raw
        .iter()
        .map(|x| (x + round_off) >> (log_q - log_p))
        .collect();

    println!("Result = [{:?}]", &h_result[0..]);

    Ok(())
}

fn create_keys(
    config: &Parameters,
    default_engine: &mut DefaultEngine,
    parallel_engine: &mut DefaultParallelEngine,
) -> Keys {
    // Create the keys
    println!("Creating keys...");
    let lwe: LweSecretKey64 = default_engine
        .generate_new_lwe_secret_key(config.n)
        .unwrap();
    let glwe: GlweSecretKey64 = default_engine
        .generate_new_glwe_secret_key(config.k, config.N)
        .unwrap();
    let bsk: LweBootstrapKey64 = parallel_engine
        .generate_new_lwe_bootstrap_key(
            &lwe,
            &glwe,
            config.Bg_bit_pbs,
            config.l_pbs,
            config.rlwe_var,
        )
        .unwrap();
    let extracted: LweSecretKey64 = default_engine
        .transform_glwe_secret_key_to_lwe_secret_key(glwe.clone())
        .unwrap();
    let ksk_extracted_lwe: LweKeyswitchKey64 = default_engine
        .generate_new_lwe_keyswitch_key(
            &extracted,
            &lwe,
            config.l_ks,
            config.base_bit_ks,
            config.rlwe_var,
        )
        .unwrap();
    println!("Keys created.");

    // Return keys struct with new keys
    Keys {
        lwe,
        glwe,
        extracted,
        ksk_extracted_lwe,
        bsk,
    }
}

fn save_keys(
    filename: &str,
    filepath: &str,
    keys: &Keys,
    serial_engine: &mut DefaultSerializationEngine,
) {
    println!("Saving keys...");

    fs::create_dir_all(filepath).unwrap();

    // Serialize the keys
    let lwe_s = serial_engine.serialize(&keys.lwe).unwrap();
    let glwe_s = serial_engine.serialize(&keys.glwe).unwrap();
    let lwe_extracted_s = serial_engine.serialize(&keys.extracted).unwrap();
    let lwe_extracted_to_lwe_ksk_s = serial_engine.serialize(&keys.ksk_extracted_lwe).unwrap();
    let lwe_bsk_s = serial_engine.serialize(&keys.bsk).unwrap();

    // Save the serialized keys into the file (ORDER MATTERS)
    let mut f = BufWriter::new(File::create(filename).unwrap());
    serialize_into(&mut f, &lwe_s).unwrap();
    serialize_into(&mut f, &glwe_s).unwrap();
    serialize_into(&mut f, &lwe_extracted_s).unwrap();
    serialize_into(&mut f, &lwe_extracted_to_lwe_ksk_s).unwrap();
    serialize_into(&mut f, &lwe_bsk_s).unwrap();

    println!("Keys saved.");
}

fn load_keys(filename: &str, serial_engine: &mut DefaultSerializationEngine) -> Keys {
    println!("Loading keys...");

    // Read into vectors (which are owned) (ORDER MATTERS)
    let mut f = BufReader::new(File::open(filename).unwrap());
    let lwe_s: Vec<u8> = deserialize_from(&mut f).unwrap();
    let glwe_s: Vec<u8> = deserialize_from(&mut f).unwrap();
    let lwe_extracted_s: Vec<u8> = deserialize_from(&mut f).unwrap();
    let lwe_extracted_to_lwe_ksk_s: Vec<u8> = deserialize_from(&mut f).unwrap();
    let lwe_bsk_s: Vec<u8> = deserialize_from(&mut f).unwrap();

    // Deserialize into keys
    let lwe: LweSecretKey64 = serial_engine.deserialize(&lwe_s[..]).unwrap();
    let glwe: GlweSecretKey64 = serial_engine.deserialize(&glwe_s[..]).unwrap();
    let extracted: LweSecretKey64 = serial_engine.deserialize(&lwe_extracted_s[..]).unwrap();
    let ksk_extracted_lwe: LweKeyswitchKey64 = serial_engine
        .deserialize(&lwe_extracted_to_lwe_ksk_s[..])
        .unwrap();
    let bsk: LweBootstrapKey64 = serial_engine.deserialize(&lwe_bsk_s[..]).unwrap();

    println!("Keys loaded.");

    Keys {
        lwe,
        glwe,
        extracted,
        ksk_extracted_lwe,
        bsk,
    }
}

fn print_key_info(keys: &Keys) {
    println!("The useful size of `lwe` is {}", size_of_val(&keys.lwe));
    println!(
        "The useful size of `glwe_key` is {}",
        size_of_val(&keys.glwe)
    );
    println!("The useful size of `bsk` is {}", size_of_val(&keys.bsk));
    println!(
        "The useful size of `extracted` is {}",
        size_of_val(&keys.extracted)
    );
    println!(
        "The useful size of `ksk_extracted_lwe` is {}",
        size_of_val(&keys.ksk_extracted_lwe)
    );
}

// fn init_keys(save_keys: bool, config: concrete::Config) -> (ClientKey, ServerKey) {
//     let keys: ClientKey;
//     let server_keys: ServerKey;
//     if (save_keys) {
//         (keys, server_keys) = generate_keys(config);

//         let mut f = BufWriter::new(File::create("./keys/keys").unwrap());
//         serialize_into(&mut f, &keys).unwrap();
//         serialize_into(&mut f, &server_keys).unwrap();
//     } else {
//         let mut f = BufReader::new(File::open("./keys/keys").unwrap());
//         keys = deserialize_from(&mut f).unwrap();
//         server_keys = deserialize_from(&mut f).unwrap();
//     }
//     return (keys, server_keys);
// }

// fn test_depth() -> Result<(), Box<dyn std::error::Error>> {
//     let config = ConfigBuilder::all_disabled().enable_default_uint3().build();

//     let save_keys = false;
//     let (keys, server_keys) = init_keys(save_keys, config);

//     set_server_key(server_keys);

//     let clear_a = 1_i64;
//     let clear_b = 1_i64;

//     let mut a = FheUint3::try_encrypt(clear_a, &keys)?;
//     let b = FheUint3::try_encrypt(clear_b, &keys)?;

//     let mut depth = 0;
//     loop {
//         a = a * &b; // Clear equivalent computations: 15 * 27 mod 256 = 149
//         let dec_a: u8 = a.decrypt(&keys);
//         if dec_a != 1u8 {
//             break;
//         }
//         depth += 1;
//     }

//     println!("depth = {}", depth);

//     Ok(())
// }
