/* HOW MANY COINS ARE IN A DEPOSIT OF MONEY
 * William Wadsworth
 * Created: 
 * Doctored: 10.25.2023
 * 
 * 
 * [DESCRIPTION]:
 * This program prompts the user to input a price, and the program will 
 * calculate and output the minimum amount of coins for each type (quarter,
 * dime, nickel, penny) required to meet the price.
 * 
 * Note: does not always work, 21.31 does not include last penny
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ coinTotal.cpp -Wall -o coinTotal
 * 
 * To run:
 *     ./coinTotal
 * 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - invalid amount
*/

#include <iostream>
#include <iomanip>
using namespace std;

int main ()
{
    // define variables
    int q, d, n, p;
    double total;

    // show 2 decimal places when outputting
    cout << fixed << showpoint << setprecision(2);
    // prompt for starting amount, store input in variable
    cout << "How much money do you have: $";
    cin >> total;

    // check amount, must be > 0
    if(total <= 0) {
        cerr << "error: amount must be greater than 0.\n";
        exit(1);
    }

    // QUARTERS
    // how many quarters in starting amount
    q = total / 0.25;
    // calculate new total without number of quarters
    total -= q*0.25;

    // DIMES
    // how many dimes in updated amount
    d = total / 0.10;
    total -= d*0.10;

    // NICKELS
    // how many nickels in updated amount
    n = total / 0.05;
    total -= n*0.05;

    // PENNIES
    // how many pennies in remaining amount
    p = total / 0.01;

    // output
    cout << "You can have as low as: " << q+d+n+p << " coins" << endl;
    cout << setw(3) << "# of quarters: " << q << endl;
    cout << setw(3) << "# of dimes: " << d << endl;
    cout << setw(3) << "# of nickels: " << n << endl;
    cout << setw(3) << "# of pennies: " << p << endl;

    return 0;
}
