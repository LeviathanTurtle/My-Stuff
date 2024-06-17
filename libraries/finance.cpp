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
#include <iomanip>      // setprecision
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
        //std::cerr << "error: amount must be greater than 0.\n";
        throw std::invalid_argument("invalid total, amount must be greater than 0");
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


/* function to generate an investment table
 * pre-condition: principal amount, interest rate, interest rate change, length of time, and
 *                deposit must all be initialized to positive non-zero floats
 * 
 * post-condition: the table is output detailing the total amount invested and the value of the
 *                 investment for each time step
*/
void finance::genInvestmentTable(const double& principal_amount, double& interest_rate, const double interest_rate_change=0, const double& time, const double& deposit)
{
    // ensure all numerical params are of valid types
    // principal 
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(principal_amount)>>::value, "principal amount must be a float type");
    // interest rate
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(interest_rate)>>::value, "Interest rate must be a float type");
    // time
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(principal_amount)>>::value, "Changing deposit amount must be a float type");
    // deposit
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(deposit)>>::value, "Deposit amount must be a float type");
    // changing interest
    static_assert(std::is_arithmetic<std::remove_reference_t<decltype(interest_rate_change)>>::value, "Changing interest rate must be a float type");

    // ========================================================================
    // TABLE

    // convert time (in years) to months
    int time_months = static_cast<int>(time) * 12;
    // loop control var
    int t = 1;
    // var to hold current investment value
    double value_of_investment = FLT_MIN;
    // var to keep track of the month for changing interest
    int count = 0;

    // show 2 decimal places
    std::cout << std::fixed << std::showpoint << std::setprecision(2);

    std::cout << std::setw(35) << "Investment Table\n\n"/* << std::endl << std::endl*/;
    std::cout << "  Month  |  Total Invested ($)  | Value of Investment ($) \n";
    std::cout << "----------------------------------------------------------\n";

    // table for changing interest
    do {
        // A = p + (p*r*t) + (t*d)
        value_of_investment = principal_amount + (principal_amount*interest_rate*time_months) + (time_months*deposit);
        std::cout << std::setw(5) << time_months << std::setw(5) << "|" << std::setw(14) 
                  << time_months*deposit << std::setw(9) << "|" << std::setw(16) 
                  << value_of_investment << "\n";
        t++;

        // for changing interest
        if (interest_rate_change != 0) {
            count++;
            if (count % 12 == 0)
                interest_rate += interest_rate_change;
        }
    } while (t <= time_months);

    std::cout << "------------------------------------------------------\n\n"/* << endl << " " << endl*/;
    std::cout << "Your capital gain will be $" << value_of_investment - principal_amount << " in " << time_months/12 << " years\n" << std::endl;
}

