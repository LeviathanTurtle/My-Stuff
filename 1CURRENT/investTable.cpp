/* INVESTMENT CALCULATOR
 * William Wadsworth
 * Created: 10.14.2020
 * Doctored: at some point (maybe)
 * CSC 1710
 * ~/csc1710/prog2/prog2.cpp
 * 
 * This program creates an investment table based on your input. Interest is compounded monthly.
 * All monetary values must be rounded to the nearest hundredth. All percentage values must be
 * rounded to the nearest thousandth and be in decimal form. All year values must be rounded to the
 * nearest whole.
 * 
 * Usage: 
 * To compile: g++ investTable.cpp -Wall -o <exe name>
 * To run: ./<exe name>
*/

#include <iostream>
#include <iomanip>
using namespace std;
int main ()
{
    // TITLE, INPUT PARAMETERS

    cout << "This program creates an investment table based on your input. "
         << "Interest is compounded monthly.\n\n";
    cout << "All monetary values must be rounded to the nearest hundredth.\n";
    cout << "All percentage values must be rounded to the nearest thousandth "
         << "and be in decimal form.\n";
    cout << "All year values must be rounded to the nearest whole." << endl;

    // ========================================================================
    // PRINCIPLE 

    double p;
    cout << "What is your principal amount: $";
    cin >> p;

    // input validation
    while (p < 0) {
        cout << "Not valid. You can't have negative money. Try again: ";
        cin >> p;
    }

    // ========================================================================
    // INTEREST RATE

    double r; // apr
    cout << "What is your annual interest rate (APR; %) as a decimal: ";
    cin >> r;

    // input validation
    while (r < 0) {
        cout << "Not valid. Negative interest rate? No, try again: ";
        cin >> r;
    }

    // ========================================================================
    // TIME

    int y, t = 1;
    cout << "How many years: ";
    cin >> y;
    // convert years to months
    y *= 12;

    // input validation
    while (y < 0) {
        cout << "Not valid. Hmm yes negative time :(. Try again: ";
        cin >> y;
    }

    // ========================================================================
    // DEPOSIT

    double d;
    cout << "How much would you like to deposit monthly: $";
    cin >> d;

    // input validation
    while (d < 0) {
        cout << "Not valid. Deposit must be greater than 0. Try again: ";
        cin >> d;
    }

    // ========================================================================
    // CHANGING INTEREST

    int cr;
    string inp;
    cout << "Does the interest change per year? ";
    cin >> inp;
    if (inp == "yes") {
        cout << "By how much (e.g. +[input]% per year): ";
        cin >> cr;
    }

    cout << "Calculating ..." << endl; 

    // ========================================================================
    // TABLE

    double A;
    int count = 0;

    // show 2 decimal places
    cout << fixed << showpoint << setprecision(2);

    // table for changing interest
    if (inp == "yes") {
        cout << setw(35) << "Investment Table" << endl << " " << endl;
        cout << "  Month  |  Total Invested ($)  | Value of Investment ($) " << endl;
        cout << "----------------------------------------------------------" << endl;

        while (t <= y) {
            A = p + (p*r*t) + (t*d);
            cout << setw(5) << t << setw(5) << "|" << setw(13) << t*d << setw(10) << "|" << setw(16) << A << endl;
            t++;
            count++;
            if (count % 12 == 0)
                r += cr;
        }
    }
    // table for constant interest
    else {
        cout << setw(35) << "Investment Table" << endl << " " << endl;
        cout << "  Month  |  Total Invested ($)  | Value of Investment ($) " << endl;
        cout << "----------------------------------------------------------" << endl;

        while (t <= y) {
	         A = p + (p*r*t) + (t*d);
            cout << setw(5) << t << setw(5) << "|" << setw(14) << t*d << setw(9) << "|" << setw(16) << A << endl;
            t++;
        }
    }

    cout << "------------------------------------------------------" << endl << " " << endl;
    cout << "Your capital gain will be $" << A - p << " in " << y/12 << " years" << endl << " " << endl;

    return 0;
}








