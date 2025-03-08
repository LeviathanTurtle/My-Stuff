/* TEMPERATURE CONVERSION CHART -- V.Rust
 * William Wadsworth
 * CSC1710
 * Created: 1.10.2020
 * Rust-ified: 3.20.2024
 * 
 * This program creates a temperature conversion chart based on a degree given in Fahrenheit,
 * incrementing by a value imput by the user.
 * 
 * Usage: 
 * To compile: rustc tempConv.rs
 * To run: ./tempConv
*/

use std::io::{self, Write};
use std::process;

//fn main() -> io::Result<()> 
fn main() {
    // introduction
    println!("This program creates a temperature conversion chart based on a degree given in \
             Fahrenheit, incrementing by a value you choose.\nAll values must be rounded to the \
             nearest thousandth.\n");

    // confirmation
    let mut confirmation = String::new();

    print!("Do you want to run this program? [Y/n]: ");
    io::stdout().flush().expect("Failed to flush stdout");
    loop { // sort of not really like cpp do-while
        confirmation.clear(); // clear input before reading again
        io::stdin().read_line(&mut confirmation).expect("Failed to read response");

        match confirmation.trim().to_lowercase().as_str() {
            "y" | "yes" => {
                break;
            }
            "n" | "no" => {
                println!("Exiting...");
                process::exit(0);
            }
            _ => { // invalid value
                print!("Invalid input. Please enter 'Y' or 'n': ");
                io::stdout().flush().expect("Failed to flush stdout");
            }
        }
    }

    // --- SMALLLEST DEGREE ---------------------
    let mut input = String::new();
    let smallest_degree: f64;

    // give instructions to define sdegree
    print!("Give your starting (smallest) Fahrenheit degree between -1000 and 1000: ");
    io::stdout().flush().expect("Failed to flush stdout");
    loop {
        input.clear();
        io::stdin().read_line(&mut input).expect("Failed to read degree");

        // input validation
        match input.trim().parse::<f64>() {
            Ok(value) if (-1000.0..=1000.0).contains(&value) => {
                smallest_degree = value;
                break;
            }
            _ => {
                print!("Not valid, degree limitations: [-1000 < this_degree < 1000]: ");
                io::stdout().flush().expect("Failed to flush stdout");
            }
        }
    }

    // --- LARGEST DEGREE -----------------------
    // same thing as smallest_degree
    let largest_degree: f64;

    print!("{}", format!("Give your ending (largest) Fahrenheit degree between {} and 1000: ", smallest_degree));
    io::stdout().flush().expect("Failed to flush stdout");
    loop {
        input.clear();
        io::stdin().read_line(&mut input).expect("Failed to read degree");

        match input.trim().parse::<f64>() {
            Ok(value) if (smallest_degree..=1000.0).contains(&value) => {
                largest_degree = value;
                break;
            }
            _ => {
                print!("{}", format!("Not valid, degree limitations: [{} < this_degree < 1000]: ", smallest_degree));
                io::stdout().flush().expect("Failed to flush stdout");
            }
        }
    }

    // --- INCREMENT ----------------------------
    // guess what? same thing again
    let increment: f64;

    print!("How much do you want to increment by: ");
    io::stdout().flush().expect("Failed to flush stdout");
    loop {
        input.clear();
        io::stdin().read_line(&mut input).expect("Failed to read increment");

        match input.trim().parse::<f64>() {
            Ok(value) if value > 0.0 && value < (largest_degree-smallest_degree) => {
                increment = value;
                break;
            }
            Ok(_) => { // successful parsing, but invalid value
                print!("{}", format!("Invalid, increment must be (0 < increment < {}): ", largest_degree-smallest_degree));
                io::stdout().flush().expect("Failed to flush stdout");
            }
            Err(_) => { // failure case
                print!("{}", format!("Invalid, increment must be (0 < increment < {}): ", largest_degree-smallest_degree));
                io::stdout().flush().expect("Failed to flush stdout");
            }
        }
    }

    // --- TABLE AND FORMULAS -------------------
    // make table heading using spaces
    println!(" Fahrenheit (°F) |  Celsius (°C)  |   Kelvin (K)   ");
    println!("---------------------------------------------------");

    // define celsius and kelvin formulae
    //double c = ((sdegree - 32) * 5/9);
    //double k = ((sdegree - 32) * 5/9 + 273.15);

    // while loop to run through incremented degrees
    let mut current_value: f64 = smallest_degree;

    while current_value <= largest_degree {
        // display calculations
        println!(
            "{:>12.3}     |{:>12.3}    |{:>12.3}",
            current_value,
            (current_value -32.0) * 5.0 / 9.0,
            (current_value -32.0) * 5.0 / 9.0 + 273.15
        );
        current_value += increment;
    }
}

