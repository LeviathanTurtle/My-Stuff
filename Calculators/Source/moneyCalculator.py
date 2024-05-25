# MONEY CALCULATOR -- V.PY
# William Wadsworth
# Created: 8.24.2020
# Doctored: 10.12.2023
# Python-ized: 3.13.2024
# CSC 1710-02
# ~/csc1710/lab2/assignment.cpp
# 
# [SUMMARY]:
# This program prompts the user for the number of 1, 5, 10, 20, 50, and 100 dollar bills they have.
# It then calculates and outputs the total sum of money based on what the user input. 
# 
# [USAGE]:
# To run: python3 moneyCalculator.py


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
using namespace std;
"""

# --- MAIN ------------------------------------------------------------------------------
# --- BILL VARS ---------------------------------
"""
int main()
{
    int bills1, bills5, bills10, bills20, bills50, bills100;
"""

# --- USER PROMPT: $1 ---------------------------
"""
    cout << "Enter the number of $1 bills: ";

    cin >> bills1;

    if(typeid(bills1) != typeid(string)) {
        cout << "error: integer number not provided.";
        exit(1);
    }
"""
bills1 = int(input("Enter the number of $1 bills: "))
while(type(bills1) != type(int)):
    bills1 = int(input("error: input must be an integer: "))
# repeat same steps for 5, 10, 20, 50, 100

# --- USER PROMPT: $5 ---------------------------
"""
    cout << "Enter the number of $5 bills: ";
    cin >> bills5;
    if(typeid(bills5) != typeid(string)) {
        cout << "error: integer number not provided.";
        exit(1);
    }
"""
bills5 = int(input("Enter the number of $5 bills: "))
while(type(bills5) != type(int)):
    bills5 = int(input("error: input must be an integer: "))

# --- USER PROMPT: $10 --------------------------
"""
    cout << "Enter the number of $10 bills: ";
    cin >> bills10;
    if(typeid(bills10) != typeid(string)) {
        cout << "error: integer number not provided.";
        exit(1);
    }
"""
bills10 = int(input("Enter the number of $10 bills: "))
while(type(bills10) != type(int)):
    bills10 = int(input("error: input must be an integer: "))

# --- USER PROMPT: $20 --------------------------
"""
    cout << "Enter the number of $20 bills: ";
    cin >> bills20;
    if(typeid(bills20) != typeid(string)) {
        cout << "error: integer number not provided.";
        exit(1);
    }
"""
bills20 = int(input("Enter the number of $20 bills: "))
while(type(bills20) != type(int)):
    bills20 = int(input("error: input must be an integer: "))

# --- USER PROMPT: $50 --------------------------
"""
    cout << "Enter the number of $50 bills: ";
    cin >> bills50;
    if(typeid(bills50) != typeid(string)) {
        cout << "error: integer number not provided.";
        exit(1);
    }
"""
bills50 = int(input("Enter the number of $50 bills: "))
while(type(bills50) != type(int)):
    bills50 = int(input("error: input must be an integer: "))

# --- USER PROMPT: $100 -------------------------
"""
    cout << "Enter the number of $100 bills: ";
    cin >> bills100;
    if(typeid(bills100) != typeid(string)) {
        cout << "error: integer number not provided.";
        exit(1);
    }
"""
bills100 = int(input("Enter the number of $100 bills: "))
while(type(bills100) != type(int)):
    bills100 = int(input("error: input must be an integer: "))

# --- CALCULATE TOTAL ---------------------------
"""
    int total = bills1 + (5*bills5) + (10*bills10) + (20*bills20) + (50*bills50) + (100*bills100);

    cout << "You have $" << total << endl;
    return 0;
}
"""
print(f"You have ${bills1+(5*bills5)+(10*bills10)+(20*bills20)+(50*bills50)+(100*bills100)}")