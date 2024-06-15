/*
 * Author: William Wadsworth
 * Date: 6.15.2024
 *
 * About:
 *    This is the implementation file for the finance class
*/

#include "finance.h"
#include <iostream>
#include <tuple>
#include <type_traits>  // static_assert


/* function to return the minimum number of coins given a monetary value
 * pre-condition: monetary total parameter should be initialized to a non-zero and non-negative
 *                float 
 * 
 * post-condition: the minimum number of coins is returned in a tuple (quarters, dimes, nickels,
 *                 then pennies). If the total value parameter is less than or equal to 0, an error
 *                 is output and a tuple consisting of four -1s is returned
*/
std::tuple<int, int, int, int> finance::findCoinTotal(double total)
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


/* function to calculate how much money based on USD denominations (1,5,10,20,50,100)
 * pre-condition: each dollar denomination count must be initialized to a positive non-zero integer
 * 
 * post-condition: the monetary total is calculated and returned
*/
int finance::moneyCalculator(const int& count_1, const int& count_5, const int& count_10, const int& count_20, const int& count_50, const int& count_100)
{
    // ensure all params are numeric
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(count_1)>>::value, "$1 denominations must be an integer type");
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(count_5)>>::value, "$5 denominations must be an integer type");
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(count_10)>>::value, "$10 denominations must be an integer type");
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(count_20)>>::value, "$20 denominations must be an integer type");
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(count_50)>>::value, "$50 denominations must be an integer type");
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(count_100)>>::value, "$100 denominations must be an integer type");

    return count_1 + (5*count_5) + (10*count_10) + (20*count_20) + (50*count_50) + (100*count_100);
}
