# FOUR FUNCTION CALCULATOR -- V.PY
# William Wadsworth
# CSC1710
# Created: 8.27.2020
# Doctored: 10.12.2023
# Python-ized: 3.18.2024
# 
# [DESCRIPTION]:
# This program provides simple four-functions on two integers given by user input.
# 
# [USAGE]:
# To run: python3 fourFunction.py


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
using namespace std;
"""

# --- MAIN ------------------------------------------------------------------------------
# --- FIRST FACTOR ------------------------------
"""
int main()
{
    int factor1, factor2;
    string response;

    cout << "Enter first integer: ";
    cin >> factor1;
"""
factor1 = int(input("Enter first integer: "))

# --- SECOND FACTOR -----------------------------
"""
    cout << "Enter second integer: ";
    cin >> factor2;
"""
factor2 = int(input("Enter second integer: "))

# --- SELECT OPERATION --------------------------
"""
    cout << "Would you like to: add, subtract, multiply, or divide? ";
    cin >> response;

    if ( response == "add" )
        cout << "The total of " << factor1 << " and " << factor2 << " is " << factor1+factor2 << endl;
    else if( response == "subtract" )
        cout << "The difference of " << factor1 << " and " << factor2 << " is " << factor1-factor2 << endl;
    else if( response == "multiply" )
        cout << "The product of " << factor1 << " and " << factor2 << " is " << factor1*factor2 << endl;
    else if ( response == "divide" )
        cout << "The product of " << factor1 << " and " << factor2 << " is " << factor1/factor2 << endl;
    else {
        cerr << "error: operation invalid.\n";
        exit(1);
    }
}
"""
response = input("Would you like to: add, subtract, multiply, or divide? ")
# input validation
while(response != "add" and response != "subtract" and response != "multiply" and response != "divide"):
    response = input("error: invalid operation: ")

if(response == "add"):
    print(f"The total of {factor1} and {factor2} is {factor1+factor2}")
elif(response == "subtract"):
    print(f"The total of {factor1} and {factor2} is {factor1-factor2}")
elif(response == "multiply"):
    print(f"The total of {factor1} and {factor2} is {factor1*factor2}")
elif(response == "divide"):
    print(f"The total of {factor1} and {factor2} is {factor1/factor2}")
# note: this else clause should not be hit due to the input validation above
else:
    print("error: operation invalid")
    exit(1)