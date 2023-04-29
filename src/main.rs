#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(unused_imports)]

use colored::Colorize;
use core::time;
use std::error::Error;
use time_graph;

pub mod mnist_rnn;
pub mod speaker_rec_rnn;
pub mod test_fns;
pub mod utils;

use crate::utils::keys::*;
use mnist_rnn::*;
use speaker_rec_rnn::*;
use test_fns::*;

fn run_tests() -> Result<(), Box<dyn Error>> {
    println!("");
    println!("{}", "Beginning tests.".bold());

    // Tests

    // MULTI GPU, MULTI NUM CT TEST
    // let ct_nums = vec![8, 32, 128, 256, 512, 768, 1024, 1536, 2048];
    // let precisions = vec![6];
    // for p in precisions {
    //     for i in &ct_nums {
    //         print_test_banner!(amortized_cuda_bs_test, true, *i as usize, 10, p, &*SET10);
    //     }
    // }

    // MULTI GPU, SINGLE NUM CT TEST
    print_test_banner!(amortized_cuda_bs_test, true, 32000, 1, 6, &*SET8);

    // print_test_banner!(fft_bootstrap_woppbs_test,);

    // print_test_banner!(
    //     test_mnist_weights_import_hashmap,
    //     "/home/vele/Documents/masters/mnist_rnn/runs/202302/20230205-190604/checkpoints/hdf5/weights.hdf5"
    // );

    // print_test_banner!(
    //     populate_depopulate_lwe_ct_vector,
    //     &*SET8
    // );

    // print_test_banner!(
    //     agnes_332,
    // );

    Ok(())
}

fn run_mnist_rnn() -> Result<(), Box<dyn Error>> {
    println!("");
    println!("{}", "Beginning MNIST RNN run.".bold());
    // print_rnn_banner!(mnist_rnn, false, false, &*SET8, 6);
    Ok(())
}

fn run_speaker_rec_rnn() -> Result<(), Box<dyn Error>> {
    println!("");
    println!("{}", "Beginning SpeakerRec RNN run.".bold());
    // print_rnn_banner!(speaker_rec_rnn, false, false, &*SET8, 11);
    Ok(())
}

fn main() {
    // Do we enable timing collection?
    time_graph::enable_data_collection(true);

    match run_tests() {
        Ok(()) => println!("{}\n", "Tests completed.".bold()),
        Err(e) => {
            println!("");
            println!("{}", "ERROR!".red().bold());
            println!("{:?}", e);
            println!("{}", e);
            println!("");
        }
    };

    match run_mnist_rnn() {
        Ok(()) => println!("{}\n", "MNIST RNN run completed.".bold()),
        Err(e) => {
            println!("");
            println!("{}", "ERROR!".red().bold());
            println!("{:?}", e);
            println!("{}", e);
            println!("");
        }
    };

    match run_speaker_rec_rnn() {
        Ok(()) => println!("{}\n", "SpeakerRec RNN run completed.".bold()),
        Err(e) => {
            println!("");
            println!("{}", "ERROR!".red().bold());
            println!("{:?}", e);
            println!("{}", e);
            println!("");
        }
    };

    // Get timings logged by time_graph
    let timings = time_graph::get_full_graph();
    println!("\n{}\n", timings.as_table());

    println!("End.");
}

// NOTE: CODE FOR ENCODING THE WHOLE PRECISION RANGE (ex. -32 -> 31 for log_p=6)
// let num_cts: usize = 1 << log_p;
// let half_range = 1 << (log_p - 1);

// let inputs_raw: Vec<i64> = (-half_range..half_range).collect(); // The whole domain
// let inputs: Vec<u64> = inputs_raw
//     .iter()
//     .map(|x| (*x as u64) << (log_q - log_p))
//     .collect();

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
