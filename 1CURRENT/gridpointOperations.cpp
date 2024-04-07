/* GRIDPOINT OPERATIONS
 * William Wadsworth
 * Created: 10.22.2020
 * Dcotored: 11.2.2023
 * CSC 1710
 * ~/csc1710/lab10/wadsworthlab10.cpp
 *
 * [DESCRIPTION]:
 * This function prompts the user for two coordinate points, and the operation
 * the user would like to perform.
 *
 * [COMPILE/RUN]:
 * To compile: g++ gridpointOperations -Wall -o gridpointOperations
 * To run: ./gridpointOperations
*/


#include <iomanip>
#include <cmath>
#include <iostream>
using namespace std;


// global variable
const double PI = 3.141592;


// function prototypes
double distance(double q, double w, double e, double r);
//double distance(double t, double y);
double radius(double a, double s, double d, double f);
//double radius(double g, double h);
double circumference(double r);
double area(double r);


int main()
{
   // local variables
   double x1, y1, x2, y2;
   string inp;

   // input requirements
   cout << "Points must be entered as two integers, such as: x x" << endl;

   // first point
   cout << "Input your first point: ";
   cin >> x1 >> y1;

   // second point
   cout << "Input your second point: ";
   cin >> x2 >> y2;

   // output decimals, ask for intended operation
   cout << fixed << showpoint << setprecision(3);
   cout << "Possible operations: distance, radius, circumference, area" << endl;
   cout << "What operation would you like to do: ";
   cin >> inp;

   // input validation
   while (inp != "distance" && inp != "radius" && inp != "circumference" && inp != "area") {
      cout << "Not valid: ";
      cin >> inp;
   }


   // call functions based on input
   if (inp == "distance")
      cout << "The distance between the two points is " << distance(x1, y1, x2, y2) << " units" << endl;
   else if (inp == "radius")
      cout << "The radius is " << radius(x1, y1, x2, y2) << " units" << endl;
   else if (inp == "circumference")
      cout << "The circumference is " << circumference(radius(x1, y1, x2, y2)) << endl;
   else if (inp == "area")
      cout << "The area is " << area(radius(x1, y1, x2, y2)) << " units squared" << endl;
   else {
      cerr << "Not valid. exiting...\n";
      exit(1);
   }

   // terminate on success, scream at me if not
   return 0;
}


// functions return calculated value depending on operation
double distance(double q, double w, double e, double r)
{
   return sqrt(pow(e - q, 2) + pow(r - w, 2));
}
/*
double distance(double t, double y)
{
   return sqrt(pow(t, 2) + pow(y, 2));
}
*/
double radius(double a, double s, double d, double f)
{
   double r = distance(a, s, d, f);

   return r;
}
/*
double radius(double g, double h)
{
   double r = distance(g, h);

   return r;
}
*/
double circumference(double r = 1)
{
   return 2 * PI * r;
}

double area(double r = 1)
{
   return PI * pow(r, 2);
}


