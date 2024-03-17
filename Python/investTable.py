# INVESTMENT CALCULATOR -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.14.2020
# Python-ized: 3.16.2024
#  
# This program creates This program creates an investment table based on your input. 
# Interest is compounded monthly. All monetary values must be rounded to the nearest hundredth.
# All percentage values must be rounded to the nearest thousandth and be in decimal form. All year
# values must be rounded to the nearest whole.


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <iomanip>
using namespace std;
"""

# --- MAIN ------------------------------------------------------------------------------
# --- INTRODUCTION ------------------------------
"""
int main ()
{
    cout << "This program creates an investment table based on your input. "
         << "Interest is compounded monthly.\n\n";
    cout << "All monetary values must be rounded to the nearest hundredth.\n";
    cout << "All percentage values must be rounded to the nearest thousandth "
         << "and be in decimal form.\n";
    cout << "All year values must be rounded to the nearest whole." << endl;
"""
print('''This program creates This program creates an investment table based on your input. 
Interest is compounded monthly. All monetary values must be rounded to the nearest hundredth.
All percentage values must be rounded to the nearest thousandth and be in decimal form. All year
values must be rounded to the nearest whole.''')

# --- PRINCIPLE ---------------------------------
"""
    double p;
    cout << "What is your principal amount: $";
    cin >> p;

    while (p < 0) {
        cout << "Not valid. You can't have negative money. Try again: ";
        cin >> p;
    }
"""
p = int(input("What is your principal amount: $"))
# input validation
while(p < 0):
    p = int(input("error: must be at least 0: $"))

# --- INTEREST RATE -----------------------------
"""
    double r; // apr
    cout << "What is your annual interest rate (APR; %) as a decimal: ";
    cin >> r;

    while (r < 0) {
        cout << "Not valid. Negative interest rate? No, try again: ";
        cin >> r;
    }
"""
apr = float(input("What is your annual interest rate (APR; %) as a decimal: "))
# input validation
while(apr < 0):
    apr = float(input("error: cannot have a negative interest rate: "))
###################### FLAG ######################

# --- TIME --------------------------------------
"""
    int y, t = 1;
    cout << "How many years: ";
    cin >> y;
    y *= 12;

    while (y < 0) {
        cout << "Not valid. Hmm yes negative time :(. Try again: ";
        cin >> y;
    }
"""
y = int(input("How many years: "))
# convert years to months
y *= 12
# input validation
while(y < 0):
    y = int(input("error: time frame must be greater than 0: "))

# --- DEPOSIT -----------------------------------
"""
    double d;
    cout << "How much would you like to deposit monthly: $";
    cin >> d;

    while (d < 0) {
        cout << "Not valid. Deposit must be greater than 0. Try again: ";
        cin >> d;
    }
"""
d = float(input("How much would you like to deposit monthly: $"))
# input validation
while(d < 0):
    d = float(input("error: deposit must be greater than 0: "))

# --- CHANGING INTEREST -------------------------
"""
    int cr;
    string inp;
    cout << "Does the interest change per year? ";
    cin >> inp;
    if (inp == "yes") {
        cout << "By how much (e.g. +[input]% per year): ";
        cin >> cr;
    }

    cout << "Calculating ..." << endl; 
"""
inp = input("Does the interest rate change per year? [Yes/No]: ")
if(inp == "Yes"):
    cr = int(input("By how much (e.g. +[input]% per year): "))
print("Calculating...\n")

# --- TABLE -------------------------------------
"""
    double A;
    int count = 0;

    cout << fixed << showpoint << setprecision(2);

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
"""
# table for changing interest
if(inp == "Yes"):
    print(f"{'Investment Table':>35}")
    print("  Month  |  Total Invested ($)  | Value of Investment ($) ")
    print("----------------------------------------------------------")
    
    for t in range(1,y+1):
        A = p + (p*apr*t) + (t*d)
        print(f"{t:>7} {t*d:>14} {A:>20}")
        if (t % 12 == 0):
            apr += cr
# table for constant interest
else:
    print(f"{'Investment Table':>35}")
    print("  Month  |  Total Invested ($)  | Value of Investment ($) ")
    print("----------------------------------------------------------")

    for t in range(1,y+1):
        A = p + (p * apr * t) + (t * d)
        print(f"{t:>7} {t*d:>14} {A:>20}")

print("------------------------------------------------------")
print(f"Your capital gain will be ${A-p:.2f} in {y//12} years")
