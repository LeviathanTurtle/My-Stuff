# IS ONE INTEGER A MULTIPLE OF THE OTHER -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.11.2020
# Doctored: 10.12.2023
# 
# Python-ized: 3.16.2024
# Updated 8.18.2024: PEP 8 Compliance
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


from sys import argv, exit


def main():
    # --- CHECK CLI -----------------------------
    if len(argv) < 3:
        print("Usage: python3 isMultiple.py <X> <Y>")
        exit(1)

    # --- SETUP VARS ----------------------------
    # convert to integer
    x = int(argv[1])
    y = int(argv[2])

    count: int = 0

    # --- DO THING ------------------------------
    # check if x is divisible by y
    if x%y == 0:
        # calculation of multiples
        for sum in range(0,x+1,y):
            print(sum)
            count += 1
        
        # output results
        print(f"\n{x} has {count} multiples of {y}")
    else:
        print(f"{x} has no multiples of {y}")


if __name__ == "__main__":
    main()