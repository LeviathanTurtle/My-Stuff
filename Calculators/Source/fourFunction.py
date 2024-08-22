# FOUR FUNCTION CALCULATOR -- V.PY
# William Wadsworth
# CSC1710
# Created: 8.27.2020
# Doctored: 10.12.2023
# 
# Python-ized: 3.18.2024
# Updated 8.18.2024: PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program provides simple four-functions on two integers given by user input.
# 
# [USAGE]:
# To run: python3 fourFunction.py


from sys import exit


def main():
    # --- FIRST FACTOR --------------------------
    factor1 = int(input("Enter first integer: "))

    # --- SECOND FACTOR -------------------------
    factor2 = int(input("Enter second integer: "))

    # --- SELECT OPERATION ----------------------
    response = input("Would you like to: add, subtract, multiply, or divide? ")
    # input validation
    while response not in ("add", "subtract", "multiply", "divide"):
        response = input("error: invalid operation: ")

    if response == "add":
        print(f"The total of {factor1} and {factor2} is {factor1+factor2}")
    elif response == "subtract":
        print(f"The total of {factor1} and {factor2} is {factor1-factor2}")
    elif response == "multiply":
        print(f"The total of {factor1} and {factor2} is {factor1*factor2}")
    elif response == "divide":
        print(f"The total of {factor1} and {factor2} is {factor1/factor2}")
    # note: this else clause should not be hit due to the input validation above
    else:
        print("error: operation invalid")
        exit(1)


if __name__ == "__main__":
    main()