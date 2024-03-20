/* TEMPERATURE CONVERSION CHART
 * William Wadsworth
 * CSC1710
 * Created: 1.10.2020
 * Rust-ified: 3.20.2024
 * 
 * This program creates a temperature conversion chart based on a degree given
 * in Fahrenheit, incrementing by a value imput by the user.
*/

// --- IMPORTS --------------------------------------------------------------------------
/*
#include <iostream>
#include <iomanip>
using namespace std;
*/
use std::io;

// --- MAIN -----------------------------------------------------------------------------
// --- INTRODUCTION -----------------------------
/*
int main ()
{
    cout << "This program creates a temperature conversion chart based on a "
         << "degree given in Fahrenheit, incrementing by a value you choose. "
         << "\nAll values must be rounded to the nearest thousandth.\n";
*/
//fn main() -> io::Result<()> 
fn main() {
    println!("This program creates a temperature conversion chart based on a degree given in \
             Fahrenheit, incrementing by a value you choose.\nAll values must be rounded to the \
             nearest thousandth.\n");

// --- CONFIRMATION -----------------------------
/*
    cout << "Do you want to run this program? [Y/n]: ";
    char confirmation;
    cin >> confirmation;

    if(confirmation == 'n') {
        cout << "terminating...\n";
        exit(0);
    }
*/
    println!("Do you want to run this program? [Y/n]: ");
    let mut confirmation = char;
    io::stdin().read_line(&mut confirmation).expect("Failed to read response");




}
    
    // display 3 decimal places in output
    cout << fixed << showpoint << setprecision(3);

    //=========================================================================
    // SMALLEST DEGREE

    // give instructions to define sdegree
    double sdegree;
    cout << "Give your starting (smallest) Fahrenheit degree [-1000 < degree <"
         << " 1000]: ";
    cin >> sdegree;

    // input validation
    while (sdegree < -1000) {
        cout << "Not valid, degree must be > -1000: ";
        cin >> sdegree;
    }

    //=========================================================================
    // LARGEST DEGREE

    // give instructions to define ldegree
    double ldegree;
    cout << "Give your ending (largest) Fahrenheit degree [-1000 < degree < "
         << "1000]: ";
    cin >> ldegree;

    // input validation
    while (ldegree < sdegree || ldegree > 1000) {
        cout << "Not valid, degree must be < 1000: ";
        cin >> ldegree;
    }

    //=========================================================================
    // INCREMENT

    // define increment
    double increment;
    cout << "How much do you want to increment by: ";
    cin >> increment;

    // input validation
    while (increment <= 0) {
        cout <<"Not valid, increment must be > 0: ";
        cin >> increment;
    }

    //=========================================================================
    // TABLE + FORMULAS

    // make table heading using spaces
    cout << " Fahrenheit (°F) |  Celsius (°C)  |   Kelvin (K)   " << endl;
    cout << "---------------------------------------------------" << endl;

    // define celsius and kelvin formulae
    //double c = ((sdegree - 32) * 5/9);
    //double k = ((sdegree - 32) * 5/9 + 273.15);

    // while loop to run through incremented degrees
    while (sdegree <= ldegree) {
        // display calculations
        cout << setw(12) << sdegree << "     |" << setw(12)  
             << ((sdegree-32) * 5/9) << "    |" << setw(12) 
             << ((sdegree-32) * 5/9 + 273.15) << endl;
        sdegree += increment;
    }

    return 0;
}
