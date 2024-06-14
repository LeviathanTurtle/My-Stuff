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
        double a, b, c;
        double point_x, point_y, point_z;

    public:
        void set_a(const double&);
        void set_b(const double&);
        void set_c(const double&);
        double get_a();
        double get_b();
        double get_c();
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
        T fourFunction(const T&, const T&, const std::string&);

        double distance(const double&, const double&, const double&, const double&);
        double radius(const double&, const double&, const double&, const double&);
        double circumference(const double&);
        double area_circle(const double&);
};

#endif

