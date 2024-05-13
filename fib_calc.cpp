// calculates the nth finonacci nunber

/* FIBONACCI CALCULATOR
 * William Wadsworth
 * Created: at some point
 * 
 * [DESCRIPTION]:
 * This program calculates the nth fibonacci number, where n is determined by the user.
 * 
 * [USAGE]:
 * To compile:
 *     g++ fib_calc.cpp -Wall -o <exe name>
 * To run:
 *     ./<exe name> [-d] < n >
 * where:
 * [-d] - optional, enable debug output
 * < n >- amount of fibonacci numbers to calculate (e.g. calculate the nth fibonacci number)
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed full execution
 * 
 * 1 - 
*/

#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{
    // check CLI args
    if(argc != 3) {
        cerr << "Error: invalid arguments. Usage: ./<exe name> [-d] <config file name> \n";
        exit(1);
    }

    

    return 0;
}