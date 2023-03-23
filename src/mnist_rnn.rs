#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use concrete_core::prelude::*;
use hdf5::{Dataset, File};
use ndarray::*;
use time_graph::{instrument, spanned};

use std::collections::HashMap;
use std::time::Instant;

use std::error::Error;

use crate::utils::common::*;
use crate::utils::datasets::*;
use crate::utils::init::*;
use crate::utils::keys::*;
use crate::utils::luts::*;
use crate::utils::layers::*;

#[macro_export]
macro_rules! print_rnn_banner {
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

fn mnist_weights_import_hashmap(
    filename: &str,
    default_engine: &mut DefaultEngine,
) -> Result<HashMap<String, Array2<i8>>, Box<dyn Error>> {
    // Open the HDF5 file and get the datasets for the weight matrices
    println!("Loading MNIST RNN weights into hashmap<name, array>.");
    println!("Loading from {}", filename);
    let file = hdf5::File::open(filename)?;

    // Create a HashMap to store the weight matrices
    let mut weight_matrices: HashMap<String, Array2<i8>> = HashMap::new();

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
        // println!("{:?}", name);
        if name.contains("quantized") {
            let parts: Vec<&str> = name.split("/").collect();
            let last_two = format!("{}/{}", parts[parts.len() - 2], parts[parts.len() - 1]);
            let mut data: Vec<i8> = dataset.read_raw()?;
            // let mut data: Vec<Cleartext64> = data.iter().map(|x| {
            //     default_engine.create_cleartext_from(x).unwrap()
            // }).collect();
            let shape = dataset.shape();
            let array: Array2<i8> =
                Array::from_shape_vec((shape[0] as usize, shape[1] as usize), data)?;
            weight_matrices.insert(last_two, array);
        }
    }

    // Print the weight matrices in the HashMap
    println!("Loaded weights.");
    // for (name, matrix) in weight_matrices.iter() {
    //     println!("{}:\n{:?}", name, matrix);
    // }

    Ok(weight_matrices)
}

#[instrument]
pub fn mnist_rnn(
    run_pt: bool,
    run_ct: bool,
    config: &Parameters,
    precision: i32,
) -> Result<(), Box<dyn Error>> {
    // Create the necessary engines
    // Here we need to create a secret to give to the unix seeder, but we skip the actual secret creation
    const UNSAFE_SECRET: u128 = 1997;
    let (
        mut default_engine,
        mut serial_engine,
        mut parallel_engine,
        mut cuda_engine,
        mut amortized_cuda_engine,
    ) = init_engines(UNSAFE_SECRET)?;

    // Create keys
    // let h_keys: Keys = create_keys(config, &mut default_engine, &mut parallel_engine)?;
    // save_keys("./keys/keys.bin", "./keys/", &h_keys, &mut serial_engine)?;
    let h_keys: Keys = load_keys("./keys/keys.bin", &mut serial_engine)?;
    let d_keys: CudaKeys = get_cuda_keys(&h_keys, &mut cuda_engine)?;
    println!("{:?}", config);

    // Establish precision
    let log_q: i32 = 64;
    let log_p: i32 = precision;
    let round_off: u64 = 1u64 << (log_q - log_p - 1);

    // Import dataset
    let mnist_config: MNISTConfig = MNISTConfig {
        mnist_images_file:
            "/data/dev/masters/tf_speaker_rec/mnist_preprocessed/mnist_images_norm_tern.npy",
        mnist_labels_file: "/data/dev/masters/tf_speaker_rec/mnist_preprocessed/mnist_labels.npy",
    };
    let (mut x, mut y): (ndarray::ArrayD<i8>, ndarray::ArrayD<i8>) = import_mnist(&mnist_config)?;
    let mut x = x.into_dimensionality::<Ix3>()?;
    let mut y = y.into_dimensionality::<Ix1>()?;

    // Import weights
    let weights: HashMap<String, Array2<i8>> = mnist_weights_import_hashmap(
        "/home/vele/Documents/masters/mnist_rnn/runs/202302/20230205-190604/checkpoints/hdf5/weights.hdf5", // 6-bit, 92%
        // "/home/vele/Documents/masters/mnist_rnn/runs/202303/20230309-102046/checkpoints/hdf5/weights.hdf5", // 6-bit, 92%, L1 out
        // "/home/vele/Documents/masters/mnist_rnn/runs/202302/20230205-174932/checkpoints/hdf5/weights.hdf5", // any-bit, 95%
        &mut default_engine
    )?;

    println!("\n==================================================\n");

    // Loop through dataset (one epoch)
    let mut correct_preds = 0;
    let mut pt_correct_preds = 0;
    let mut dense_out_dif_percent: Vec<f32> = vec![];
    let mut dense_out_mae: Vec<f32> = vec![];
    let dense_out_num_accs = 4;
    let num_test_images = 100;
    for (i, img) in x.axis_iter(ndarray::Axis(0)).enumerate() {
        
        // Stopping condition
        if i == num_test_images {
            break;
        }

        println!("Running sample {} ------------------------------------------------------------------", i+1);

        // Encrypt inputs
        let pt = img.clone().to_owned().mapv(|x| x as i32);
        let ct = encrypt_lwe_array(&img, log_p, log_q, &h_keys.extracted, &config, &mut default_engine)?;

        // -------------------------- START ENCRYPTED FWD STEP ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        println!("Beginning encrypted run.");

        // RNN BLOCK 0 --------------------------------------------------------------------------------------------------------
        let (qrnn_0, pt_qrnn_0) = spanned!("qrnn_0", {
            encrypted_rnn_block(
                &ct.view(),
                &pt.view(),
                &weights["QRNN_0/quantized_kernel:0"].view(),
                &weights["QRNN_0/quantized_recurrent_kernel:0"].view(),
                "QRNN_0",
                log_p, log_q,
                &d_keys,&h_keys, config,
                &mut cuda_engine,&mut amortized_cuda_engine,&mut default_engine,
            )?
        });
        // println!("Finished encrypted QRNN_0.");
        // ---------------------------------------------------------------------------------------------------------------------

        // TIME REDUCTION LAYER ------------------------------------------------------------------------------------------------
        let tr = time_reduction(qrnn_0.view())?;
        let pt_tr = time_reduction(pt_qrnn_0.view())?;
        // println!("Finished encrypted TR.");
        // ---------------------------------------------------------------------------------------------------------------------

        // RNN BLOCK 1 ---------------------------------------------------------------------------------------------------------
        let (qrnn_1, pt_qrnn_1) = spanned!("qrnn_1", {
            encrypted_rnn_block(
                &tr,
                &pt_tr,
                &weights["QRNN_1/quantized_kernel:0"].view(),
                &weights["QRNN_1/quantized_recurrent_kernel:0"].view(),
                "QRNN_1",
                log_p, log_q,
                &d_keys, &h_keys, config,
                &mut cuda_engine, &mut amortized_cuda_engine, &mut default_engine,
            )?
        });
        // println!("Finished encrypted QRNN_1.");
        // ---------------------------------------------------------------------------------------------------------------------

        // FLATTEN -------------------------------------------------------------------------------------------------------------
        let flattened = flatten_2D(qrnn_1.view())?;
        let pt_flattened = flatten_2D(pt_qrnn_1.view())?;
        // println!("Finished encrypted FLATTEN.");
        // ---------------------------------------------------------------------------------------------------------------------

        // DENSE BLOCK 0 -------------------------------------------------------------------------------------------------------
        let (dense_0, pt_dense_0) = spanned!("dense_0", {
            encrypted_dense_block(
                &flattened,
                &pt_flattened,
                &weights["DENSE_0/quantized_kernel:0"].view(),
                "DENSE_0",
                true,
                1,
                log_p, log_q,
                &d_keys, &h_keys, config,
                &mut cuda_engine, &mut amortized_cuda_engine, &mut default_engine,
            )?
        });
        // println!("Finished encrypted DENSE_0.");
        // ---------------------------------------------------------------------------------------------------------------------

        // DENSE BLOCK 1 -------------------------------------------------------------------------------------------------------
        let (dense_out, pt_dense_out) = spanned!("dense_out", {
            encrypted_dense_block(
                &dense_0.view(),
                &pt_dense_0.view(),
                &weights["DENSE_OUT/quantized_kernel:0"].view(),
                "DENSE_OUT",
                false,
                dense_out_num_accs,
                log_p, log_q,
                &d_keys, &h_keys, config,
                &mut cuda_engine, &mut amortized_cuda_engine, &mut default_engine,
            )?
        });
        // check_pt_ct_difference(&dense_out.view(), &pt_dense_out.view(), format!("{}: output", "DENSE_OUT").as_str(), false, log_p, log_q, &h_keys, &mut default_engine)?;
        // println!("Finished encrypted DENSE_OUT.");
        // ---------------------------------------------------------------------------------------------------------------------

        // Decrypt, convert to signed
        let mut ct_logits: Array2<u64> = decrypt_lwe_array(
            &dense_out.view(),
            log_p,
            log_q,
            &h_keys.extracted,
            &mut default_engine,
        )?;
        let mut ct_logits = ct_logits.mapv(|x| iP_to_iT::<i32>(x, log_p));
        let mut ct_logits = ct_logits.sum_axis(Axis(0));
        let pt_dense_out = pt_dense_out.into_shape(ct_logits.dim())?;

        // Calculate some stats
        let dense_out_stats = check_pt_pt_difference(&ct_logits.view(), &pt_dense_out.view(), format!("{}: output", "DENSE_OUT").as_str(), false)?;
        dense_out_dif_percent.push(dense_out_stats.0);
        dense_out_mae.push(dense_out_stats.1);

        // Get result
        let ct_result = compute_softmax_then_argmax(&ct_logits)?;
        println!("Completed encrypted run.");

        

        // -------------------------- END ENCRYPTED FWD STEP ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // ======================================================================================================================================================================================================================================
        // ======================================================================================================================================================================================================================================
        // ======================================================================================================================================================================================================================================

        // -------------------------- BEGIN PLAINTEXT FWD STEP ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        // {

        //     println!("Beginning plaintext run.");

        //     // RNN BLOCK 0 --------------------------------------------------------------------------------------------------------
        //     let pt_qrnn_0 = plaintext_rnn_block(
        //         &pt.view(),
        //         &weights["QRNN_0/quantized_kernel:0"].mapv(|x| x as i32).view(),
        //         &weights["QRNN_0/quantized_recurrent_kernel:0"].mapv(|x| x as i32).view()
        //     )?;
        //     // ---------------------------------------------------------------------------------------------------------------------

        //     // TIME REDUCTION LAYER ------------------------------------------------------------------------------------------------
        //     let pt_tr = time_reduction(pt_qrnn_0.view())?;
        //     // ---------------------------------------------------------------------------------------------------------------------

        //     // RNN BLOCK 1 ---------------------------------------------------------------------------------------------------------
        //     let pt_qrnn_1 = plaintext_rnn_block(
        //         &pt_tr,
        //         &weights["QRNN_1/quantized_kernel:0"].mapv(|x| x as i32).view(),
        //         &weights["QRNN_1/quantized_recurrent_kernel:0"].mapv(|x| x as i32).view()
        //     )?;
        //     // ---------------------------------------------------------------------------------------------------------------------

        //     // FLATTEN -------------------------------------------------------------------------------------------------------------
        //     let pt_flattened = flatten_2D(pt_qrnn_1.view())?;
        //     // ---------------------------------------------------------------------------------------------------------------------

        //     // DENSE BLOCK 0 -------------------------------------------------------------------------------------------------------
        //     let pt_dense_0 = plaintext_dense_block(
        //         &pt_flattened,
        //         &weights["DENSE_0/quantized_kernel:0"].mapv(|x| x as i32).view(),
        //         true
        //     )?;
        //     // ---------------------------------------------------------------------------------------------------------------------

        //     // DENSE BLOCK 1 -------------------------------------------------------------------------------------------------------
        //     let pt_dense_out = plaintext_dense_block(
        //         &pt_dense_0.view(),
        //         &weights["DENSE_OUT/quantized_kernel:0"].mapv(|x| x as i32).view(),
        //         false
        //     )?;
        //     // ---------------------------------------------------------------------------------------------------------------------

        // }

        let pt_result = compute_softmax_then_argmax(&pt_dense_out)?;
        println!("Completed plaintext run.\n");

        println!("Encrypted MNIST RNN result: {}", ct_result);
        println!("Plaintext MNIST RNN result: {}", pt_result);
        println!("True result:                {}\n", y[[i]]);

        // Metric calculations
        if ct_result as i8 == y[[i]] {
            correct_preds += 1;
        }
        if pt_result as i8 == y[[i]] {
            pt_correct_preds += 1;
        }
        println!("Correct CT predictions = {}", correct_preds);
        println!("Correct PT predictions = {}\n", pt_correct_preds);
    }

    // Stat calculations
    let acc = 100_f32 * correct_preds as f32 / num_test_images as f32;
    let pt_acc = 100_f32 * pt_correct_preds as f32 / num_test_images as f32;
    println!("\nCompleted {} predictions!", num_test_images);
    println!("\nAccuracy Statistics...");
    println!("CT Accuracy = {:.2}%", acc);
    println!("PT Accuracy = {:.2}%", pt_acc);

    let dense_out_dif_percent = arr1(&dense_out_dif_percent);
    let dense_out_mae = arr1(&dense_out_mae);
    println!("\nDENSE_OUT Statistics...");
    println!("Number of accumulators                 = {}", dense_out_num_accs);
    println!("Percent different elements (mean, std) = ({:.2}%, {:.2}%)", dense_out_dif_percent.mean().unwrap(), dense_out_dif_percent.std(0f32));
    println!("MAE (mean, std)                        = ({:.2}, {:.2})", dense_out_mae.mean().unwrap(), dense_out_mae.std(0f32));

    Ok(())
}

// ORIGINAL CODE FOR RNN BLOCK
// let num_units = weights["QRNN_0/quantized_kernel:0"].dim().1;
// let (mut d_output, mut d_temp, d_luts) = prepare_layer(log_p, log_q, num_units, config, &mut cuda_engine, &mut default_engine)?;
// let mut states: Array2<LweCiphertext64> = Array2::from_shape_vec((0, num_units), vec![])?;
// for (t, ct_t) in ct.rows().into_iter().enumerate() {
//     // W_x * x
//     let mut output = matmul_custom_1D(&ct_t, &weights["QRNN_0/quantized_kernel:0"].view(), &config, &mut default_engine)?;

//     if t != 0 { // First state is zeros but I don't want to initialize zeros
//         // W_h * h
//         let mut hidden = matmul_custom_1D(&states.row(t-1) , &weights["QRNN_0/quantized_recurrent_kernel:0"].view(), &config, &mut default_engine)?;

//         // W_x * x + W_h * h
//         for (x_i, h_i) in output.iter_mut().zip(hidden.iter()) {
//             default_engine.fuse_add_lwe_ciphertext(x_i, h_i)?;
//         }
//     }

//     // sign(W_x * x + W_h * h)
//     let mut output = sign_activation(output, &mut d_temp, &mut d_output, &d_luts, &d_keys, &mut cuda_engine, &mut amortized_cuda_engine, &mut default_engine)?;

//     states.append(Axis(0), output.view().into_shape((1, num_units))?)?;

//     // println!("{:?}", timestep);
// };

// CODE TO PRINT OUT IMAGES, THEIR TRUE VALUE, ETC.
// ct.mapv_inplace(|c| {
//     let mut out = c.clone();
//     default_engine.discard_add_lwe_ciphertext(&mut out, &c, &c).unwrap();
//     out
// });
// if i == 0 {
//     let mut raw = decrypt_lwe_array(&ct.view(), log_p, log_q, &h_keys.extracted, &mut default_engine)?;
//     // println!("{:?}\n", ct);
//     // println!("{:?}\n", raw);
//     // println!("{:?}\n", img);
//     for j in 0..28 {
//         println!("");
//         for k in 0..28 {
//             print!("{}", raw[[j, k]]);
//         }
//     }
//     print!("         {}\n", y[[i]]);
//     for j in 0..28 {
//         println!("");
//         for k in 0..28 {
//             print!("{}", img[[j, k]]);
//         }
//     }
//     print!("         {}\n", y[[i]]);
//     println!("Correctness: {:?}", raw == img);
//     println!("Number of different elements = {}", count_different_elts(&raw.view(), &img.view()));
//     break;
// }

// TEST DATASET
// for b in 0..10 {
//     println!("\n");
//     for i in 0..28 {
//         println!("");
//         for j in 0..28 {
//             print!("{}", x[[b, i, j]])
//         }
//     }
//     print!("         {}", y[[b]])
// }
