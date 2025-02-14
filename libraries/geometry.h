/*
 * Author: William Wadsworth
 * Date: 6.14.2024
 *
 * About:
 *    This is the header file for the C++ geometry class
*/

#ifndef GEOMETRY
#define GEOMETRY


class geometry {
    private:
        double a, b, c;
        double angle_a, angle_b, angle_c;
        double point_x, point_y, point_z;

        double cir_radius, cir_diameter = 2*cir_radius, cir_circumference;

    public:
        void set_a(const double&);
        void set_b(const double&);
        void set_c(const double&);
        double get_a();
        double get_b();
        double get_c();

        void set_angle_a(const double&);
        void set_angle_b(const double&);
        void set_angle_c(const double&);
        double get_angle_a();
        double get_angle_b();
        double get_angle_c();

        void set_point_x(const double&);
        void set_point_y(const double&);
        void set_point_z(const double&);
        double get_point_x();
        double get_point_y();
        double get_point_z();

        void set_radius(const double&);
        double get_radius();

        void set_diameter(const double&);
        double get_diameter();

        void set_circumference(const double&);
        double get_circumference();

        double distance(const double&, const double&, const double&, const double&);
        double radius(const double&, const double&, const double&, const double&);
        double circumference(const double&);
        double area_circle(const double&);
        
        bool isRightTriangle(const double&, const double&, const double&);
};

#endif

