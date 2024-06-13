/*
 * Author: William Wadsworth
 * Date: 6.12.2024
 *
 * About:
 *    This is the implementation file for the calculator class
*/

#include "calculator.h"
#include <iostream>
#include <tuple>


/* function to return the minimum number of coins given a monetary value
 * pre-condition: monetary total parameter should be initialized to a non-zero and non-negative
 *                float 
 * 
 * post-condition: the minimum number of coins is returned in a tuple (quarters, dimes, nickels,
 *                 then pennies). If the total value parameter is less than or equal to 0, an error
 *                 is output and a tuple consisting of four -1s is returned
*/
std::tuple<int, int, int, int> calculator::findCoinTotal(double total)
{
    // check amount, must be > 0
    if(total <= 0) {
        std::cerr << "error: amount must be greater than 0.\n";
        return std::make_tuple(-1,-1,-1,-1);
    }

    // convert dollars to cents to avoid floating-point issues
    total = static_cast<int>(total * 100 + 0.5); // adding 0.5 to round correctly

    // QUARTERS
    // how many quarters in starting amount
    int q = total / 25;
    // calculate new total without number of quarters
    total -= q * 25;

    // DIMES
    // how many dimes in updated amount
    int d = total / 10;
    total -= d * 10;

    // NICKELS
    // how many nickels in updated amount
    int n = total / 5;
    total -= n * 5;

    // PENNIES
    // how many pennies in remaining amount
    int p = total;

    return std::make_tuple(q,d,n,p);
}


/* function to return the factorial or double factorial of a number
 * pre-condition: endpoint should be initialized to an integer between 0 and 1,000 and
 *                double_factorial must be initialized to true of false
 * 
 * post-condition: if the user's endpoint is valid, its calculated result is returned, otherwise -1
 *                 is returned
*/
long int calculator::factorial(const int& endpoint, const bool& double_factorial)
{
    // check that the user's endpoint is valid. Will probably update this later to be more dynamic
    if (endpoint < 0 || endpoint > 1000) {
        std::cerr << "Not valid, endpoint integer must be between 0 and 1,000\n";
        return -1;
    }

    long int prod = INT_MIN;

    // user wants a double factorial
    if (double_factorial) {
        // additional check unique to double factorials
        if (endpoint%2 == 0) {
            std::cerr << "Not valid, endpoint integer must be odd and between 0 and 1,000\n";
            return -1;
        }

        for (int x=1; x <= endpoint; x++)
            prod *= x;

    } else { // normal factorial
        int x = 1;

        do {
            prod *= x;
            x += 2;
        } while (x <= endpoint);
    }

    return prod;
}


/* function to return the geometric series of a number
 * pre-condition: 
 * 
 * post-condition: 
*/
double calculator::geoseries(double a, const int& num_terms, double r=0.5)
{
    // check common ratio of choice
    // TODO: re-check this
    if (r >= 0)
        r = 0.5;

    double sum = FLT_MIN;

    for (int i=0; i<num_terms; i++) {
        sum += a;
        a *= r;
    }

    return sum;
}


/* function to 
 * pre-condition: 
 * 
 * post-condition: 
*/



/* function to 
 * pre-condition: 
 * 
 * post-condition: 
*/



/* function to 
 * pre-condition: 
 * 
 * post-condition: 
*/
