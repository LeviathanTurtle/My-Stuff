# HOW MANY COINS ARE IN A DEPOSIT OF MONEY
# William Wadsworth
# Created: 
# Doctored: 10.25.2023
# Python-ized: 4.23.2024
# 
# [DESCRIPTION]:
# This program prompts the user to input a price, and the program will calculate and output the
# minimum amount of coins for each type (quarter, dime, nickel, penny) required to meet the price.
# 
# Note: does not always work, 21.31 does not include last penny. This will be fixed in the future.
# 
# [USAGE]:
# To run: python3 coinTotal.py
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed a full execution
# 
# 1 - invalid amount


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <iomanip>
using namespace std;
"""
import sys

# --- MAIN ------------------------------------------------------------------------------
# --- GET TOTAL ---------------------------------
"""
int main ()
{
    int q, d, n, p;
    double total;

    cout << fixed << showpoint << setprecision(2);
    cout << "How much money do you have: $";
    cin >> total;

    if(total <= 0) {
        cerr << "error: amount must be greater than 0.\n";
        exit(1);
    }
"""
# prompt for starting amount, store input in variable
total = float(input("How much money do you have: $"))
# input validation, must be > 0
if total <= 0:
    sys.stderr.wrte("error: amount must be greater than 0.\n")
    exit(1)

# --- QUARTERS ----------------------------------
"""
    q = total / 0.25;
    total -= q*0.25;
"""
# how many quarters in starting amount
q = int(total // 0.25)
# calculate new total without number of quarters
total = total - q*0.25
# this process is repeated for each coin

# --- DIMES -------------------------------------
"""
    d = total / 0.10;
    total -= d*0.10;
"""
# how many dimes in updated amount
d = int(total // 0.10)
total = total - d*0.10

# --- NICKELS -----------------------------------
"""
    n = total / 0.05;
    total -= n*0.05;
"""
# how many nickels in updated amount
n = int(total // 0.05)
total = total - n*0.05

# --- PENNIES -----------------------------------
"""
    p = total / 0.01;
"""
# how many pennies in remaining amount
p = int(total // 0.01)
    
# --- OUTPUT ------------------------------------
"""
    cout << "You can have as low as: " << q+d+n+p << " coins" << endl;
    cout << setw(3) << "# of quarters: " << q << endl;
    cout << setw(3) << "# of dimes: " << d << endl;
    cout << setw(3) << "# of nickels: " << n << endl;
    cout << setw(3) << "# of pennies: " << p << endl;

    return 0;
}
"""
print(f"\nYou can have as low as: {q+d+n+p} coins")
print(f"{'# of quarters: ':>3} {q}")
print(f"{'# of dimes: ':>3} {d}")
print(f"{'# of nickels: ':>3} {n}")
print(f"{'# of pennies: ':>3} {p}")