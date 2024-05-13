/* FIBONACCI CALCULATOR
 * William Wadsworth
 * Created: at some point
 * 
 * [DESCRIPTION]:
 * This program calculates the nth fibonacci number, where n is determined by the user. This only
 * works with small values.
 * 
 * [USAGE]:
 * To compile:
 *     g++ fib_calc.cpp -Wall -o <exe name>
 * To run:
 *     ./<exe name> < n >
 * where:
 * < n > - amount of fibonacci numbers to calculate (e.g. calculate the nth fibonacci number)
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed full execution
 * 
 * 1 - invalid CLI args
*/

#include <iostream>
using namespace std;

int fibonacci(const long int);

int main(int argc, char* argv[])
{
    // check CLI args
    if(argc != 2) {
        cerr << "Error: invalid arguments. Usage: ./<exe name> [-d] <config file name> \n";
        exit(1);
    }

    cout << fibonacci(atoi(argv[1])) << endl;

    return 0;
}

int fibonacci(const long int n)
{
    if (n <= 1)
        return n;

    return fibonacci(n-1) + fibonacci(n-2);
}

