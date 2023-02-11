#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

use colored::Colorize;
use std::error::Error;

pub mod test_fns;
pub mod utils;

use test_fns::*;

fn run_tests() -> Result<(), Box<dyn Error>> {
    println!("");
    println!("{}", "Beginning tests.".bold());

    // Tests
    for i in 0..12 {
        print_test_banner!(amortized_cuda_bs_test, false, 2_i32.pow(i) as usize, 50);
    }

    Ok(())
}

fn main() {
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
