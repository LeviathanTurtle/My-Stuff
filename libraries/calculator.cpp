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
#include <numbers>       // std::numbers::pi


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


// function to update the value of point_x
void calculator::set_point_x(const double& new_point_x)
{
    point_x = new_point_x;
}
// function to return the current point_x value
double calculator::get_point_x()
{
    return point_x;
}


// function to update the value of point_y
void calculator::set_point_y(const double& new_point_y)
{
    point_y = new_point_y;
}
// function to return the current point_y value
double calculator::get_point_y()
{
    return point_y;
}


// function to update the value of point_z
void calculator::set_point_z(const double& new_point_z)
{
    point_z = new_point_z;
}
// function to return the current point_z value
double calculator::get_point_z()
{
    return point_z;
}


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
        throw std::invalid_argument("invalid total");
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
        throw std::invalid_argument("invalid endpoint");
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
    if (r >= 0)
        r = 0.5;

    double sum = FLT_MIN;

    for (int i=0; i<num_terms; i++) {
        sum += a;
        a *= r;
    }

    return sum;
}


/* function that returns the sum, difference, product, or quotient of two numbers (supports 
 * integers and floats). 
 * pre-condition: operand_1 and operand_2 parameters must be initialized with values. If dividing,
 *                operand_2 cannot be 0. operation parameter must be initialized to a non-empty
 *                string
 * 
 * post-condition: depending on the operation specified (assuming the operation is valid), the sum,
 *                 difference, product, or quotient is returned, otherwise an error is output and a
 *                 relevant exception is thrown
*/
template <typename T>
T calculator::fourFunction(const T& operand_1, const T& operand_2, const std::string& operation)
{
    // convert string to lowercase
    std::transform(operation.begin(), operation.end(), operation.begin(), ::tolower);

    if (operation == "add")
        return operand_1 + operand_2;
    else if (operation == "subtract")
        return operand_1 - operand_2;
    else if (operation == "multiply")
        return operand_1 * operand_2;
    else if (operation == "divide")
        if (operand_2 == 0) {
            std::cerr << "Error: cannot divide by 0\n";
            throw std::invalid_argument("cannot divide by 0");
        } else
            return operand_1 / operand_2;
    else {
        std::cerr << "Error: invalid operation\n";
        throw std::invalid_argument("invalid operation");
    }
}


/* function to calculate the distance between two cartesian coordinates (x1,y1,x2,y2)
 * pre-condition: the coordinates (passed as (x1,y1,x2,y2)) should be initialized as valid
 *                cartesian coordinate points
 * 
 * post-condition: the distance between the two points is returned
*/
// todo: double check this
double calculator::distance(const double& q, const double& w, const double& e, const double& r)
{
    return sqrt(pow(e-q,2) + pow(r-w,2));
}


/* function to calculate the radius between two cartesian coordinates (x1,y1,x2,y2)
 * pre-condition: the coordinates (passed as (x1,y1,x2,y2)) should be initialized as valid
 *                cartesian coordinate points 
 * 
 * post-condition: the radius between the two points is returned
*/
// todo: double check this
double calculator::radius(const double& a, const double& s, const double& d, const double& f)
{
    return distance(a,s,d,f);
}


/* function to determine the circumference of a circle
 * pre-condition: the radius must be initialized to a positive non-zero float
 * 
 * post-condition: the calculated circumference is returned
*/
double calculator::circumference(const double& radius)
{
    return 2*std::numbers::pi * radius;
}


/* function to calculate the area of a circle
 * pre-condition: the radius must be initialized to a positive non-zero float
 * 
 * post-condition: the calculated area is returned
*/
double calculator::area_circle(const double& radius)
{
    return std::numbers::pi * pow(radius, 2);
}


/* function to 
 * pre-condition: 
 * 
 * post-condition: 
*/





