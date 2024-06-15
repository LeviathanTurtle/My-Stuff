/*
 * Author: William Wadsworth
 * Date: 6.15.2024
 *
 * About:
 *    This is the header file for the finance class
*/

#ifndef FINANCE
#define FINANCE

#include <tuple>


class finance {
    private:
        int x;

    public:
        std::tuple<int, int, int, int> findCoinTotal(double);
        int moneyCalculator(const int&, const int&, const int&, const int&, const int&, const int&);
};

#endif

