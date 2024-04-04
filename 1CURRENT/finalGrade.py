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


int test()
{
    //lab variables
    double lab1 = 100.0, lab2 = 95.0, lab3 = 80.0, lab4 = 85.0;

    //quiz variables
    double quiz1 = 85.0, quiz2 = 70.0, quiz3 = 75.0;

    //program variable
    double prog = 91.0;

    //test variable
    double test = 87.5;

    //calculation 1: lab and quiz averages
    double avglab = (lab1 + lab2 + lab3 + lab4)/4;
    double avgquiz = (quiz1 + quiz2 + quiz3)/3;

    //calculation 2: final grade
    double fin = (.1*avglab + .1*avgquiz + .3*prog + .3*test) / .8;
    cout << "Your final grade is " << fin << "%" << endl;

}

int lab3b()
{
    //define lab variables
    double lab1, lab2, lab3, lab4;
    
    //ask for input, store in variable; lab1
    //cout << "Enter your first lab grade: ";
    cin >> lab1;

    //repeat with remaining lab variables
    //cout << "Enter your second lab grade: ";
    cin >> lab2;

    //cout << "Enter your third lab grade: ";
    cin >> lab3;

    //cout << "Enter your fourth lab grade: ";
    cin >> lab4;

    //define quiz variables
    double quiz1, quiz2, quiz3;

    //ask for input, store in variable; quiz1
    //cout << "Enter your first quiz grade: ";
    cin >> quiz1;

    //repeat with remaining quiz variables
    //cout << "Enter your second quiz grade: ";
    cin >> quiz2;

    //cout << "Enter your third quiz grade: ";
    cin >> quiz3;

    //define program variable
    double prog;

    //ask for input, store in variable; program
    //cout << "Enter your program grade: ";
    cin >> prog;

    //define test variable
    double test;

    //ask for input, store in variable; test
    //cout << "Enter your test grade: ";
    cin >> test;

    //calculate avg quiz and lab
    double avgquiz = (10*lab1 + 10*lab2 + 10*lab3 + 10*lab4)/4;
    double avglab = (10*quiz1 + 10*quiz2 + 10*quiz3)/3;

    //final calculation
    double fin = (.1*avgquiz + .1*avglab + .3*prog + .3*test)/.8;
    cout << "Your final grade is " << fin << "%" << endl;
}

