# TEMPERATURE CONVERSION CHART -- V.PY
# William Wadsworth
# CSC1710
# Created: 1.10.2020
# Python-ized: 3.11.2024
# 
# This program creates a temperature conversion chart based on a degree given
# in Fahrenheit, incrementing by a value imput by the user.


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <iomanip>
#using namespace std;
"""

# --- MAIN ------------------------------------------------------------------------------
# --- INTRODUCTION ------------------------------
"""
int main ()
{
    cout << "This program creates a temperature conversion chart based on a "
         << "degree given in Fahrenheit, incrementing by a value you choose. "
         << "\nAll values must be rounded to the nearest thousandth.\n";
"""
print('''This program creates a temperature convrsion chart based on a degree given in
Fahrenheit, incrementing by a value you choose.\nAll values must be rounded to the nearest thousandth.''')

# --- CONFIRMATION ------------------------------
"""
    cout << "Do you want to run this program? [Y/n]: ";
    char confirmation;
    cin >> confirmation;
    if(confirmation == 'n') {
        cout << "terminating...\n";
        exit(0);
    }
"""
confirmation = input("Do you want to run this program? [Y/n]: ")
# check confirmation
while confirmation != 'Y' and confirmation != 'n':
    confirmation = input("Please enter [Y/n]: ")
# if declined, terminate
if confirmation == 'n':
    print('terminating...')
    exit(0)

# --- SET UP OUTPUT -----------------------------
"""
    cout << fixed << showpoint << setprecision(3);
"""
# instead of being declared functionally global, it will be specified when needed

# --- SMALLEST DEGREE -------------------------------------------------------------------
"""
    double sdegree;
    cout << "Give your starting (smallest) Fahrenheit degree [-1000 < degree <"
         << " 1000]: ";
    cin >> sdegree;

    while (sdegree < -1000) {
        cout << "Not valid, degree must be > -1000: ";
        cin >> sdegree;
    }
"""
sdegree = int(input("Give your starting (smallest) Fahrenheit degree [-1000 < this_degree < 1000]: "))
# input validation
while(sdegree < -1000 or sdegree > 1000):
    sdegree = int(input("Not valid, degree limitations: [-1000 < this_degree < 1000]: "))

# --- LARGEST DEGREE --------------------------------------------------------------------
"""
    double ldegree;
    cout << "Give your ending (largest) Fahrenheit degree [-1000 < degree < "
         << "1000]: ";
    cin >> ldegree;

    while (ldegree < sdegree || ldegree > 1000) {
        cout << "Not valid, degree must be < 1000: ";
        cin >> ldegree;
    }
"""
ldegree = int(input("Give your ending (largest) Fahrenheit degree [smallest_degree < this_degree < 1000]: "))
# input validation
while(ldegree <= sdegree or ldegree > 1000):
    ldegree = int(input("Not valid, degree limitations: [smallest_degree < this_degree < 1000]: "))

# --- INCREMENT -------------------------------------------------------------------------
"""
    double increment;
    cout << "How much do you want to increment by: ";
    cin >> increment;

    while (increment <= 0) {
        cout <<"Not valid, increment must be > 0: ";
        cin >> increment;
    }
"""
increment = float(input("How much do you want to increment by: "))
# input validation
while(increment <= 0):
    increment = float(input("Not valid, increment must be > 0: "))

# --- TABLE + FORMULAS ------------------------------------------------------------------
"""
    cout << " Fahrenheit (°F) |  Celsius (°C)  |   Kelvin (K)   " << endl;
    cout << "---------------------------------------------------" << endl;

    while (sdegree <= ldegree) {
        cout << setw(12) << sdegree << "     |" << setw(12)  
             << ((sdegree-32) * 5/9) << "    |" << setw(12) 
             << ((sdegree-32) * 5/9 + 273.15) << endl;
        sdegree += increment;
    }
"""
print(''' Fahrenheit (°F) |  Celsius (°C)  |   Kelvin (K)   
---------------------------------------------------''')

# celsius and kelvin formulae:
# float c = ((sdegree -32) * 5/9)
# float k = ((sdegree -32) * 5/9 + 273.15)
tmp1 = ' '

# while loop to run through incremented degrees
while(sdegree <= ldegree):
    # display calculations
    print(f"{sdegree:12.3f}     |{((sdegree - 32) * 5 / 9):12.3f}    |{((sdegree - 32) * 5 / 9 + 273.15):12.3f}")
   #print(f"{sdegree:12.3f} |{tmp1:3}{((sdegree - 32) * 5 / 9):12.3f}|{tmp1:3}{((sdegree - 32) * 5 / 9 + 273.15):12.3f}")
    sdegree += increment



    #return 0;
#}
