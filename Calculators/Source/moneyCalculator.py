# MONEY CALCULATOR -- V.PY
# William Wadsworth
# Created: 8.24.2020
# Doctored: 10.12.2023
# 
# Python-ized: 3.13.2024
# Updated 8.18.2024: function decomposition and PEP 8 Compliance
# 
# CSC 1710-02
# ~/csc1710/lab2/assignment.cpp
# 
# [SUMMARY]:
# This program prompts the user for the number of 1, 5, 10, 20, 50, and 100 dollar bills they have.
# It then calculates and outputs the total sum of money based on what the user input. 
# 
# [USAGE]:
# To run: python3 moneyCalculator.py


# pre-condition: prompt must be a valid string
# post-condition: returns a valid integer entered by the user
def get_input(prompt: str) -> int:
    """Prompt the user for an integer input and validate it."""
    
    # repeat until valid input is given
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Error: Input must be an integer.")


def main():
    # --- USER PROMPT ---------------------------
    bills1 = get_input("Enter the number of $1 bills: ")
    bills5 = get_input("Enter the number of $5 bills: ")
    bills10 = get_input("Enter the number of $10 bills: ")
    bills20 = get_input("Enter the number of $20 bills: ")
    bills50 = get_input("Enter the number of $50 bills: ")
    bills100 = get_input("Enter the number of $100 bills: ")

    # --- CALCULATE TOTAL -----------------------
    print(f"You have ${bills1+(5*bills5)+(10*bills10)+(20*bills20)+(50*bills50)+(100*bills100)}")


if __name__ == "__main__":
    main()