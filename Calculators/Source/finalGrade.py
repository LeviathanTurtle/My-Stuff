# FINAL GRADE CALCULATOR -- V.PY
# William Wadsworth
# Created: at some point
# Doctored: at some other point
# 
# Python-ized: 3.18.2024
# Updated 8.18.2024: PEP 8 Compliance
# 
# CSC 1710 or 1720 idk but probably 10
#  
# This program calculates a final grade of a class based on 4 labs, 3 quizzes, and one program and
# test grade. The percentages are fixed, but can be adjusted in the code below.
#
# Usage: python3 finalGrade.py


def main():
    # --- LABS ----------------------------------
    lab1 = float(input("Enter your first lab grade: "))
    lab2 = float(input("Enter your second lab grade: "))
    lab3 = float(input("Enter your third lab grade: "))
    lab4 = float(input("Enter your fourth lab grade: "))
    print("\n")

    # --- QUIZZES -------------------------------
    quiz1 = float(input("Enter your first quiz grade: "))
    quiz2 = float(input("Enter your second quiz grade: "))
    quiz3 = float(input("Enter your third quiz grade: "))
    print("\n")

    # --- PROGRAM -------------------------------
    prog = float(input("Enter your program grade: "))
    print("\n")

    # --- TEST --------------------------------------
    test = float(input("Enter your test grade: "))
    print("\n")

    # --- CALCULATION ---------------------------
    avglab = (10*(lab1+lab2+lab3+lab4))/4
    avgquiz = (10*(quiz1+quiz2+quiz3))/3
    print(f"Your final grade is {(.1*avgquiz + .1*avglab + .3*prog + .3*test)/.8}%")


if __name__ == "__main__":
    main()