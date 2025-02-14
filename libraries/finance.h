/*
 * Author: William Wadsworth
 * Date: 6.15.2024
 *
 * About:
 *    This is the header file for the C++ finance class
*/

#ifndef FINANCE
#define FINANCE

#include <tuple>


class finance {
    public:
        std::tuple<int, int, int, int> findCoinTotal(double);
        int moneyCalculator(const int&, const int&, const int&, const int&, const int&, const int&);
        void genInvestmentTable(const double&, double&, const double, const double&, const double&);
};

#endif

