# TEMPERATURE CONVERSION CHART -- V.PY
# William Wadsworth
# CSC1710
# Created: 1.10.2020
# 
# Python-ized: 3.11.2024
# Updated 8.17.2024: PEP 8 Compliance
# 
# This program creates a temperature conversion chart based on a degree given
# in Fahrenheit, incrementing by a value imput by the user.
# 
# Usage: python3 tempConv.py


from sys import exit


def main():
    # --- INTRODUCTION --------------------------
    print("""This program creates a temperature convrsion chart based on a degree given in
    Fahrenheit, incrementing by a value you choose.\nAll values must be rounded to the nearest 
    thousandth.""")

    # --- CONFIRMATION --------------------------
    confirmation = input("Do you want to run this program? [Y/n]: ")
    # check confirmation
    while confirmation not in ("Y","n"):
        confirmation = input("Please enter [Y/n]: ")
    # if declined, terminate
    if confirmation == "n":
        print('terminating...')
        exit(0)

    # --- SMALLEST DEGREE -----------------------
    start_degree = int(input("Give your starting (smallest) Fahrenheit degree [-1000 < this_degree < 1000]: "))
    # input validation
    while start_degree < -1000 or start_degree > 1000:
        start_degree = int(input("Not valid, degree limitations: [-1000 < this_degree < 1000]: "))

    # --- LARGEST DEGREE ------------------------
    end_degree = int(input("Give your ending (largest) Fahrenheit degree [smallest_degree < this_degree < 1000]: "))
    # input validation
    while end_degree <= start_degree or end_degree > 1000:
        start_degree = int(input("Not valid, degree limitations: [smallest_degree < this_degree < 1000]: "))

    # --- INCREMENT -----------------------------
    increment = float(input("How much do you want to increment by: "))
    # input validation
    while increment <= 0:
        increment = float(input("Not valid, increment must be > 0: "))

    # --- TABLE + FORMULAS ----------------------
    print(""" Fahrenheit (°F) |  Celsius (°C)  |   Kelvin (K)   
    ---------------------------------------------------""")

    # celsius and kelvin formulae:
    # float cel = ((start_degree -32) * 5/9)
    # float kel = ((start_degree -32) * 5/9 + 273.15)

    # while loop to run through incremented degrees
    current_degree = float(start_degree)
    while start_degree <= end_degree:
        # display calculations
        celsius: float = (current_degree - 32) * 5 / 9
        kelvin: float = celsius + 273.15
        print(f"{current_degree:12.2f}     |{celsius:12.2f}    |{kelvin:12.2f}")
        current_degree += increment


if __name__ == "__main__":
    main()