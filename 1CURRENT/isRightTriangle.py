# DO THE INPUT SIDES MAKE A RIGHT TRIANGLE -- V.PY
# William Wadsworth
# CSC1710
# Created: 9.28.2020
# Doctored: 10.25.2023
# Python-ized: 3.16.2024
#  
# This program takes three sides as input from the user and determines if they make a right 
# triangle.


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
using namespace std;
"""

# --- MAIN ------------------------------------------------------------------------------
# --- PROMPT ------------------------------------
"""
int main ()
{
    int a, b, c;
    
    cout << "Give 3 sides of a triangle.\n Must be rounded to the nearest "
         << "whole, > 0, and in order of a b c (e.g. 3 4 5): " << endl;
    cin >> a >> b >> c;
"""

# prompt for sides
print("Give 3 sides of a triangle.")
print("Must be rounded to the nearest whole, > 0, and in order of a b c (e.g. 3 4 5): ")
# variables for each side, store in respective variable
a, b, c = map(int, input().split())

# --- PYTHAGOREAN --------------------------------
"""
    if ((a*a) + (b*b) == (c*c))
        cout << "This is a right triangle" << endl;
    else
        cout << "This is not a right triangle" << endl;

    return 0;
}
"""
# pythagorean theorem
if((a*a) + (b*b) == (c*c)):
    print("This is a right triangle.")
else:
    print("This is not a right triangle.")