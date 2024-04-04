/* 4-FUNCTION CALCULATOR

 * William Wadsworth
 * Created: 8.27.2020
 * Doctored: 10.12.2023
 * CSC1710-02
 * ~/csc1710/lab2/lab2.cpp
 * 
 * 
 * [SUMMARY]:
 * 4-function basic calculator
 * 
 * 
 * [USE]:
 * To compile:
 *     g++ fourFunction.cpp -Wall -o fourFunction
 * 
 * To run:
 *     ./fourFunction
 * 
*/

#include <iostream>
using namespace std;

int main()
{
    // variables
    int factor1, factor2;
    string response;
    // get input1
    cout << "Enter first integer: ";
    // store first input in variable factor1
    cin >> factor1;
    
    // repeat with factor2
    cout << "Enter second integer: ";
    cin >> factor2;

    // user picks operation
    cout << "Would you like to: add, subtract, multiply, or divide? ";
    cin >> response;

    // if statements to run operations
    if ( response == "add" )
        cout << "The total of " << factor1 << " and " << factor2 << " is " << factor1+factor2 << endl;
    else if( response == "subtract" )
        cout << "The difference of " << factor1 << " and " << factor2 << " is " << factor1-factor2 << endl;
    else if( response == "multiply" )
        cout << "The product of " << factor1 << " and " << factor2 << " is " << factor1*factor2 << endl;
    else if ( response == "divide" )
        cout << "The product of " << factor1 << " and " << factor2 << " is " << factor1/factor2 << endl;
    else {
        cerr << "error: operation invalid.\n";
        exit(1);
    }
}
