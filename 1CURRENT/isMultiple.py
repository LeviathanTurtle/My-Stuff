# IS ONE INTEGER A MULTIPLE OF THE OTHER -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.11.2020
# Doctored: 10.12.2023
# Python-ized: 3.16.2024
#  
# [SUMMARY]:
# This program takes two integers X and Y and determines if X is a multiple of Y. The integers are
# passed as CLI arguments using argc and argv. There should only be 3 arguments: the exe and the 
# two integers. If X is a multiple of Y, the program will calculate and output each factor until it
# reaches X.
# 
# [USAGE]:
# To run (3 args):
#     python3 isMultiple.py <X> <Y>
# where <X> and <Y> are the integers you want to use.
# Restrictions: X and Y must be greater than 0 and rounded to the nearest whole.
#
# [EXAMPLE RUN]:
# $: python3 isMultiple.py 50 10
# 10
# 20
# 30
# 40
# 50
# 
# 50 has 5 multiples of 10
# 
# [EXIT CODES]:
# 1 - incorrect CLI argument usage


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
using namespace std;
"""
import sys

# --- MAIN ------------------------------------------------------------------------------
# --- CHECK CLI ---------------------------------
"""
int main(int argc, char* argv[])
{
    if(argc != 3) {
        cerr << "error: invalid number of arguments. " << argc << " provided.\n";
        exit(1);
    }
"""
if len(sys.argv) < 3:
    print("Usage: python3 isMultiple.py <X> <Y>")
    sys.exit(1)

# --- SETUP VARS --------------------------------
"""
    int sum=0, count=0;

    int x = atoi(argv[1]);
    int y = atoi(argv[2]);
"""
# convert char* to integer
x = int(sys.argv[1])
y = int(sys.argv[2])

count = 0

# --- DO THING ----------------------------------
"""
    if( x % y == 0) {
        while(sum < x) {
            sum += y;
            cout << sum << endl;
            count++;
        }

        cout << endl << x << " has " << count << " multiples of " << y << endl;
    }
    else
        cout << x << " has no multiples of " << y << endl;

    return 0;
}
"""
# check if x is divisible by y
if(x%y == 0):
    # calculation of multiples
    for sum in range(0,x+1,y):
        print(sum)
        count += 1
    
    # output results
    print("\n", x, " has ", count, " multiples of ", y)
else:
    print(x, " has no multiples of ", y)