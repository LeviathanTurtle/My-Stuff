# TEMPERATURE CONVERSION CHART -- V.PY
# William Wadsworth
# CSC1710
# Created: 
# Python-ized: 3.18.2024
#  
# This program 


# --- IMPORTS ---------------------------------------------------------------------------
"""
include <iostream>
using namespace std;
"""

# --- MAIN ------------------------------------------------------------------------------
# --- LABS --------------------------------------
"""
int lab3a()
{
    double lab1, lab2, lab3, lab4;
    
    cout << "Enter your first lab grade: ";
    cin >> lab1;

    cout << "Enter your second lab grade: ";
    cin >> lab2;

    cout << "Enter your third lab grade: ";
    cin >> lab3;

    cout << "Enter your fourth lab grade: ";
    cin >> lab4;
"""
lab1 = float(input("Enter your first lab grade: "))
lab2 = float(input("Enter your second lab grade: "))
lab3 = float(input("Enter your third lab grade: "))
lab4 = float(input("Enter your fourth lab grade: "))

# --- QUIZZES -----------------------------------
"""
    cout << endl;
    double quiz1, quiz2, quiz3;

    cout << "Enter your first quiz grade: ";
    cin >> quiz1;

    cout << "Enter your second quiz grade: ";
    cin >> quiz2;

    cout << "Enter your third quiz grade: ";
    cin >> quiz3;
"""
quiz1 = float(input("Enter your first quiz grade: "))
quiz2 = float(input("Enter your second quiz grade: "))
quiz3 = float(input("Enter your third quiz grade: "))

# --- PROGRAM -----------------------------------
"""
    double prog;

    cout << "Enter your program grade: ";
    cin >> prog;
"""
prog = float(input("Enter your program grade: "))

# --- TEST --------------------------------------
"""
    double test;

    cout << "Enter your test grade: ";
    cin >> test;
"""
test = float(input("Enter your test grade: "))

# --- CALCULATION -------------------------------
"""
    double avgquiz = (10*lab1 + 10*lab2 + 10*lab3 + 10*lab4)/4;
    double avglab = (10*quiz1 + 10*quiz2 + 10*quiz3)/3;

    double fin = (.1*avgquiz + .1*avglab + .3*prog + .3*test)/.8;
    cout << "Your final grade is " << fin << "%" << endl;
}
"""
avglab = (10*(lab1+lab2+lab3+lab4))/4
avgquiz = (10*(quiz1+quiz2+quiz3))/3
print(f"Your final grade is {(.1*avgquiz + .1*avglab + .3*prog + .3*test)/.8}%")

