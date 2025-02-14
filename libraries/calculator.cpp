/*
 * Author: William Wadsworth
 * Date: 6.12.2024
 *
 * About:
 *    This is the implementation file for the C++ calculator class
*/

#include "calculator.h"
#include <iostream>


// function to update the value of a
void calculator::set_a(const double& new_a)
{
    a = new_a;
}
// function to return the current a value
double calculator::get_a()
{
    return a;
}


// function to update the value of b
void calculator::set_b(const double& new_b)
{
    b = new_b;
}
// function to return the current b value
double calculator::get_b()
{
    return b;
}


// function to update the value of c
void calculator::set_c(const double& new_c)
{
    c = new_c;
}
// function to return the current c value
double calculator::get_c()
{
    return c;
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
        //std::cerr << "Not valid, endpoint integer must be between 0 and 1,000\n";
        throw std::invalid_argument("Invalid endpoint, integer must be between 0 and 1,000");
    }

    long int prod = INT_MIN;

    // user wants a double factorial
    if (double_factorial) {
        // additional check unique to double factorials
        if (endpoint%2 == 0) {
            std::cerr << "Invalid endpoint, integer must be between 0 and 1,000\n";
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
 * pre-condition: starting value a must be initialized to a non-zero float, number of terms must be
 *                initialized to a non-zero integer, r must be initialized to a positive non-zero
 *                float, but has a default of 0.5 if it is not provided or invalid
 * 
 * post-condition: the series sum is returned
*/
double calculator::geoseries(double a, const int& num_terms, double r=0.5)
{
    // check common ratio of choice
    // TODO: re-check this
    if (r <= 0)
        r = 0.5;

    double sum = 0;

    for (int i=0; i<num_terms; i++) {
        sum += a;
        a *= r;
    }

    return sum;
}

