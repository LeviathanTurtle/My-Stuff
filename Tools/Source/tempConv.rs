/* TEMPERATURE CONVERSION CHART -- V.Rust
 * William Wadsworth
 * CSC1710
 * Created: 1.10.2020
 * Rust-ified: 3.20.2024
 * 
 * This program creates a temperature conversion chart based on a degree given in Fahrenheit,
 * incrementing by a value imput by the user.
*/

use std::io;

//fn main() -> io::Result<()> 
fn main() {
    // introduction
    println!("This program creates a temperature conversion chart based on a degree given in \
             Fahrenheit, incrementing by a value you choose.\nAll values must be rounded to the \
             nearest thousandth.\n");

    // confirmation
    println!("Do you want to run this program? [Y/n]: ");
    // personal note: `mut` is required because for this way because we are defining a variable,
    // then assigning it later
    let mut confirmation = char;
    io::stdin().read_line(&mut confirmation).expect("Failed to read response");

    // --- SMALLLEST DEGREE ---------------------
    // give instructions to define sdegree
    println!("Give your starting (smallest) Fahrenheit degree [-1000 < this_degree < 1000]: ");
    let mut smallest_degree = int;
    io::stdin().read_line(&mut smallest_degree).expect("Failed to read degree");

    // input validation
    while smallest_degree <= -1000 && smallest_degree >= 1000 {
        println!("Not valid, degree limitations: [-1000 < this_degree < 1000]: ");
        io::stdin().read_line(&mut smallest_degree).expect("Failed to read degree");
    }

    // --- LARGEST DEGREE -----------------------
    // give instructions to define ldegree
    println!("Give your ending (largest) Fahrenheit degree [smallest_degree < this_degree < 1000]: ");
    let mut largest_degree = int;
    io::stdin().read_line(&mut largest_degree).expect("Failed to read degree");

    // input validation
    while largest_degree <= smallest_degree && largest_degree >= 1000 {
        println!("Not valid, degree limitations: [smallest_degree < this_degree < 1000]: ");
        io::stdin().read_line(&mut largest_degree).expect("Failed to read degree");
    }

    // --- INCREMENT ----------------------------
    // define increment
    println!("How much do you want to increment by: ");
    let mut increment = float;
    io::stdin().read_line(&mut increment).expect("Failed to read increment");

    // input validation
    while increment <= 0 {
        println!("Not valid, increment must be > 0: ");
        io::stdin().read_line(&mut increment).expect("Failed to read increment");
    }

    // --- TABLE AND FORMULAS -------------------
    // make table heading using spaces
    println!(" Fahrenheit (°F) |  Celsius (°C)  |   Kelvin (K)   ");
    println!("---------------------------------------------------");

    // define celsius and kelvin formulae
    //double c = ((sdegree - 32) * 5/9);
    //double k = ((sdegree - 32) * 5/9 + 273.15);

    // while loop to run through incremented degrees
    while(smallest_degree <= largest_degree) {
        // display calculations
        println!("{:>12.3}     |{:>12.3}    |{:>12.3}", smallest_degree, (smallest_degree - 32) * 5 / 9, (smallest_degree - 32) * 5 / 9 + 273.15);
        smallest_degree += increment;
    }
}

