# DO THE INPUT SIDES MAKE A RIGHT TRIANGLE -- V.PY
# William Wadsworth
# CSC1710
# Created: 9.28.2020
# Doctored: 10.25.2023
# 
# Python-ized: 3.16.2024
# Updated 8.18.2024: PEP 8 Compliance
#  
# This program takes three sides as input from the user and determines if they make a right 
# triangle.


def main():
    # --- PROMPT --------------------------------
    # prompt for sides
    print("""Give 3 sides of a triangle.\nMust be rounded to the nearest whole, > 0, and in order
          of a b c (e.g. 3 4 5): """)
    # variables for each side, store in respective variable
    a, b, c = map(int, input().split())

    # --- PYTHAGOREAN ---------------------------
    # pythagorean theorem
    if (a*a) + (b*b) == (c*c):
        print("This is a right triangle.")
    else:
        print("This is not a right triangle.")


if __name__ == "__main__":
    main()