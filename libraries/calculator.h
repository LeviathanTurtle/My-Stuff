/*
 * Author: William Wadsworth
 * Date: 6.12.2024
 *
 * About:
 *    This is the header file for the calculator class
*/

#ifndef CALCULATOR
#define CALCULATOR

#include <tuple>

class calculator {
    private:


    public:
        std::tuple<int, int, int, int> findCoinTotal(double);
        long int factorial(const int&, const bool&);
        double geoseries(double, const int&, double);
};

#endif

