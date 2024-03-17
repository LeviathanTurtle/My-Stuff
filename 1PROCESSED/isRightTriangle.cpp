/* DO THE INPUT SIDES MAKE A RIGHT TRIANGLE
 * William Wadsworth
 * Created: 9.28.2020
 * Doctored: 10.25.2023
 * CSC1710
 * ~/csc1710/chap4/triangle.cpp
 * 
 * do the inputed sides make a right triangle?
*/

#include <iostream>
//#include <cmath>
using namespace std;

int main ()
{
    // variables for each side
    int a, b, c;

    // prompt for sides
    cout << "Give 3 sides of a triangle.\nMust be rounded to the nearest "
         << "whole, > 0, and in order of a b c (e.g. 3 4 5): " << endl;
    // store in respective variables
    cin >> a >> b >> c;

    // pythagorean theorem
    if ((a*a) + (b*b) == (c*c))
        cout << "This is a right triangle" << endl;
    else
        cout << "This is not a right triangle" << endl;

    return 0;
}
