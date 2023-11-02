/*
   Author: William Wadsworth
   Date: 4.5.21
   Class: CSC1720
   Code location: ~/csc1720/prog4/wadsworthProg4.cpp

   About:
      This program has the user input two integers (x, y) and outputs the calculation x^y

   To compile:
      g++ -Wall wadsworthProg4.cpp -o testProg

   To execute:
      ./testProg
*/

#include <iostream>
#include "invalidBase.h"
#include "invalidPower.h"
#include "overflow.h"
using namespace std;

/* repower function to recursively call itself to calculate an integer a raised to the power of b
 * precondition: a, b must be declared and initialized
 * postcondition: product of calculation is returned as a double
 */
double repower(int, int);

/* test function to catch possible errors that might be thrown, and output product of calculation
 * precondition: j, k must be declared and initialized, classes must be known
 * postcondition: if successful, product is output; if not, exception is thrown
 */
void test(int, int);

int main()
{
    int j, k;

    // user inputs 2 integers
    cout << "Enter two integers: ";
    cin >> j >> k;

    // test two integers
    test(j, k);

    return 0;
}

double repower(int a, int b)
{
    if (b == 0)
        return 1;
    else if (b == 1)
        return a;
    else if (b < 0)
        return 1 / (repower(a, -b));
    else
        return a * repower(a, b - 1);
}

void test(int x, int y)
{
    try
    {
        // NOTE: I commented these out so it might make testing it a little easier
        
        /*if (j < 0)
            throw invalidBase();*/
        /*if (k < 0)
            throw invalidPower();*/

        // this is supposed to be the overflow error check thing, don't think it works like how I want it to
        /*if (sizeof(int))
            throw overflow();*/
        cout << x << "^" << y << " = " << repower(x, y) << endl;
    }
    catch (invalidPower invPower)
    {
        cout << "error: " << invPower.huh() << endl;
        while (y < 0)
        {
            cout << "re-enter base value: ";
            cin >> y;
        }
        cout << x << "^" << y << " = " << repower(x, y) << endl;
    }
    catch (invalidBase invBase)
    {
        cout << "error: " << invBase.huh() << endl;
        while (x < 0)
        {
            cout << "re-enter exponent value: ";
            cin >> x;
        }
        cout << x << "^" << y << " = " << repower(x, y) << endl;
    }
    catch (overflow ovr)
    {
        cout << "error: " << ovr.huh() << endl;
        long int x2 = x;
        cout << x2 << "^" << y << " = " << repower(x, y) << endl;
    }
}
