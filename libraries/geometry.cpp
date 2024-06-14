/*
 * Author: William Wadsworth
 * Date: 6.14.2024
 *
 * About:
 *    This is the implementation file for the geometry class
*/

#include "geometry.h"
#include <iostream>
#include <numbers>     // std::numbers::pi


/* begin get/set methods */

// function to update the value of a
void geometry::set_a(const double& new_a)
{
    a = new_a;
}
// function to return the current a value
double geometry::get_a()
{
    return a;
}


// function to update the value of b
void geometry::set_b(const double& new_b)
{
    b = new_b;
}
// function to return the current b value
double geometry::get_b()
{
    return b;
}


// function to update the value of c
void geometry::set_c(const double& new_c)
{
    c = new_c;
}
// function to return the current c value
double geometry::get_c()
{
    return c;
}


// function to update the value of angle_a
void geometry::set_angle_a(const double& new_angle_a)
{
    angle_a = new_angle_a;
}
// function to return the current value of angle_a
double geometry::get_angle_a()
{
    return angle_a;
}


// function to update the value of angle_b
void geometry::set_angle_b(const double& new_angle_b)
{
    angle_b = new_angle_b;
}
// function to return the current value of angle_b
double geometry::get_angle_b()
{
    return angle_b;
}


// function to update the value of angle_c
void geometry::set_angle_c(const double& new_angle_c)
{
    angle_c = new_angle_c;
}
// function to return the current value of angle_c
double geometry::get_angle_c()
{
    return angle_c;
}


// function to update the value of point_x
void geometry::set_point_x(const double& new_point_x)
{
    point_x = new_point_x;
}
// function to return the current point_x value
double geometry::get_point_x()
{
    return point_x;
}


// function to update the value of point_y
void geometry::set_point_y(const double& new_point_y)
{
    point_y = new_point_y;
}
// function to return the current point_y value
double geometry::get_point_y()
{
    return point_y;
}


// function to update the value of point_z
void geometry::set_point_z(const double& new_point_z)
{
    point_z = new_point_z;
}
// function to return the current point_z value
double geometry::get_point_z()
{
    return point_z;
}


// function to update the current radius value
void geometry::set_radius(const double& new_radius)
{
    cir_radius = new_radius;
}
// function to return the current radius value
double geometry::get_radius()
{
    return cir_radius;
}


// function to update the current diameter value
void geometry::set_diameter(const double& new_diameter)
{
    cir_diameter = new_diameter;
}
// function to return the current diameter value
double geometry::get_diameter()
{
    return cir_diameter;
}


// function to update the current circumference value
void geometry::set_circumference(const double& new_circumference)
{
    cir_circumference = new_circumference;
}
// function to return the current circumference value
double geometry::get_circumference()
{
    return cir_circumference;
}

/* end get/set methods */


/* function to calculate the distance between two cartesian coordinates (x1,y1,x2,y2)
 * pre-condition: the coordinates (passed as (x1,y1,x2,y2)) should be initialized as valid
 *                cartesian coordinate points
 * 
 * post-condition: the distance between the two points is returned
*/
// todo: double check this
double geometry::distance(const double& q, const double& w, const double& e, const double& r)
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
double geometry::radius(const double& a, const double& s, const double& d, const double& f)
{
    return distance(a,s,d,f);
}


/* function to determine the circumference of a circle
 * pre-condition: the radius must be initialized to a positive non-zero float
 * 
 * post-condition: the calculated circumference is returned
*/
double geometry::circumference(const double& radius)
{
    return 2*std::numbers::pi * radius;
}


/* function to calculate the area of a circle
 * pre-condition: the radius must be initialized to a positive non-zero float
 * 
 * post-condition: the calculated area is returned
*/
double geometry::area_circle(const double& radius)
{
    return std::numbers::pi * pow(radius, 2);
}


/* function to determine if the sides of a triangle make up a right triangle
 * pre-condition: sides parameters a, b, and c must be initialized to positive non-zero floats. The
 *                hypotenuse should be side c
 * 
 * post-condition: if the triangle is a right triangle, true is returned, otherwise false
*/
bool geometry::isRightTriangle(const double& a, const double& b, const double& c)
{
    if ((a*a) + (b*b) == (c*c)) return true;
    else return false;
}




