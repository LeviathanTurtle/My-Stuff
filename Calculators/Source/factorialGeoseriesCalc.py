# FACTORIAL AND GEOSERIES CALCULATOR -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.15.2020
# Doctored: 10.15.2023
# 
# Python-ized: 3.19.2024
# Updated 8.18.2024: PEP 8 Compliance
#  
# This program performs a factorial or geoseries calculation based on a number given from input.
# 
# Usage: python3 factorialGeoseriesCalc.py


# --- FUNCTIONS -------------------------------------------------------------------------
# --- FACTORIAL ---------------------------------
# pre-condition: endpoint must be a non-negative integer (0 <= endpoint <= 1000)
# post-condition: returns the factorial of endpoint
def factorial(endpoint: int) -> int:
    """Calculate the factorial of a given number."""
    
    prod: int = 1
    
    for x in range(1,endpoint+1):
        prod *= x
    
    return prod


# --- DOUBLE FACTORIAL --------------------------
# pre-condition: endpoint must be an odd non-negative integer (0 <= endpoint <= 1000)
# post-condition: returns the double factorial of endpoint
def double_factorial(endpoint: int) -> int:
    """Calculate the double factorial of a given odd number."""
    
    x: int = 1
    prod: int = 1
    
    while x <= endpoint:
        prod *= x
        x += 2
    
    return prod


# --- GEOSERIES ---------------------------------
# pre-condition: t must be a non-negative integer, a and r must be real numbers
# post-condition: returns the sum of the first t terms of the geometric series with first term a
#                 and common ratio r
def geoseries(a: float, t: int, r: float =0.5) -> float:
    """Calculate the sum of a geometric series."""
    
    sum: float = 0.0
    
    for _ in range(t):
        sum += a
        a *= r
        
    return sum


def main():
    # --- INTRODUCTION --------------------------
    print("Factorial and Geometric series calculations")

    # --- OPERATION SELECTION -------------------
    inp = input("Select operation 'factorial', 'dfactorial', or 'geometric': ")
    # input validation
    while inp not in ("factorial", "dfactorial", "geometric"):
        inp = input("error: invalid operation: ")

    # --- FACTORIAL -----------------------------
    if inp == "factorial":
        # parameters for input, store in variable
        print("\nInteger must be between 0 and 1,000\n")
        
        endpoint = int(input("Enter an integer to the nearest whole for factorial calculation: "))
        # input validation
        while endpoint < 0 or endpoint > 1000:
            endpoint = int(input("Not valid, integer must be between 0 and 1,000: "))
        
        # ouput calculation
        print(f"{endpoint}! = {factorial(endpoint)}")

    # --- DOUBLE FACTORIAL ----------------------
    elif inp == "dfactorial":
        # parameters for input, store in variable
        print("\nInteger must be between 0 and 1,000\n")
        
        endpoint = int(input("Enter an odd integer to the nearest whole for double factorial calculation: "))
        # input validation
        while endpoint % 2 == 0 or endpoint < 0 or endpoint > 1000:
            endpoint = int(input("Not valid, integer must be odd and between 0 and 1,000: "))
        
        # output calculation
        print(f"{endpoint}!! = {double_factorial(endpoint)}")

    # --- GEOSERIES -----------------------------
    # assume geoseries
    else:
        print("Sum of Geometric series")
        
        a = float(input("What is your first term: "))    
        t = int(input("How many terms would you like to take the sum of: "))
        # this is to avoid converting an empty string to a float
        try:
            r = float(input("Enter your common ratio of choice (leave blank for 0.5): "))
        except ValueError:
            r = 0.5
        
        print(f"Sum of {t} terms, a = {a}, r = {r}, is {geoseries(a,t,r)}")


if __name__ == "__main__":
    main()