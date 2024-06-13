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
        int a, b, c;
        int point_x, point_y, point_z;

    public:
        void set_a(const int&);
        void set_b(const int&);
        void set_c(const int&);
        int get_a();
        int get_b();
        int get_c();
        void set_point_x(const double&);
        void set_point_y(const double&);
        void set_point_z(const double&);
        double get_point_x();
        double get_point_y();
        double get_point_z();

        std::tuple<int, int, int, int> findCoinTotal(double);
        long int factorial(const int&, const bool&);
        double geoseries(double, const int&, double);
        
        template <typename T>
        T calculator::fourFunction(const T&, const T&, const std::string&);
};

#endif

