# INVESTMENT CALCULATOR -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.14.2020
# 
# Python-ized: 3.16.2024
# Updated 8.18.2024: function decomposition and PEP 8 Compliance
#  
# This program creates This program creates an investment table based on your input. 
# Interest is compounded monthly. All monetary values must be rounded to the nearest hundredth.
# All percentage values must be rounded to the nearest thousandth and be in decimal form. All year
# values must be rounded to the nearest whole.
# 
# Usage: python3 investTable.py





def main():
    # --- INTRODUCTION ------------------------------
    print("""This program creates This program creates an investment table based on your input. 
    Interest is compounded monthly. All monetary values must be rounded to the nearest hundredth.
    All percentage values must be rounded to the nearest thousandth and be in decimal form. All
    year values must be rounded to the nearest whole.""")

    # --- PRINCIPLE ---------------------------------
    p = int(input("\nWhat is your principal amount: $"))
    # input validation
    while p < 0:
        p = int(input("error: must be at least 0: $"))

    # --- INTEREST RATE -----------------------------
    apr = float(input("What is your annual interest rate (APR; %) as a decimal: "))
    # input validation
    while apr < 0:
        apr = float(input("error: cannot have a negative interest rate: "))
    ###################### FLAG ######################

    # --- TIME --------------------------------------
    y = int(input("How many years: "))
    # convert years to months
    y *= 12
    # input validation
    while y < 0:
        y = int(input("error: time frame must be greater than 0: "))

    # --- DEPOSIT -----------------------------------
    d = float(input("How much would you like to deposit monthly: $"))
    # input validation
    while d < 0:
        d = float(input("error: deposit must be greater than 0: "))

    # --- CHANGING INTEREST -------------------------
    inp = input("Does the interest rate change per year? [Yes/No]: ")
    if inp in ("Yes", "No"):
        cr = float(input("By how much (e.g. +[input]% per year): "))
    print("Calculating...\n")

    # --- TABLE -------------------------------------
    # table for changing interest
    if inp == "Yes":
        print(f"{'Investment Table':>35}")
        print("  Month  |  Total Invested ($)  | Value of Investment ($) ")
        print("----------------------------------------------------------")
        
        for t in range(1,y+1):
            a = p + (p*apr*t) + (t*d)
            print(f"{t:>6} {t*d:>14.2f} {a:>20.2f}")
            if (t % 12 == 0):
                apr += cr
    # table for constant interest
    else:
        print(f"{'Investment Table':>35}")
        print("  Month  |  Total Invested ($)  | Value of Investment ($) ")
        print("----------------------------------------------------------")

        for t in range(1,y+1):
            a = p + (p * apr * t) + (t * d)
            print(f"{t:>6} {t*d:>14.2f} {a:>20.2f}")

    print("------------------------------------------------------")
    print(f"Your capital gain will be ${a-p:.2f} in {y//12} years")


if __name__ == "__main__":
    main()