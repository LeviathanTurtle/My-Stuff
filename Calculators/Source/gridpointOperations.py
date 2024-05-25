# GRIDPOINT OPERATIONS -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.22.2020
# Dcotored: 11.2.2023
# Python-ized: 3.17.2024
# 
# [DESCRIPTION]:
# This program prompts the user for two coordinate points, and the operation the user would like to
# perform. In the context of ellipsoids, this program assumes you are working with a circle.
# 
# [USAGE]:
# To run: python3 gridpointOperations.py


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iomanip>
#include <cmath>
#include <iostream>
using namespace std;
"""
from math import sqrt

# --- GLOBAL VARS -----------------------------------------------------------------------
"""
const double PI = 3.141592;
"""
PI = 3.141592

# --- FUNCTION DEFINITIONS --------------------------------------------------------------
# --- DISTANCE ----------------------------------
"""
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
"""
# functions return calculated value depending on operation
def distance(q, w, e = None, r = None) -> float:
    #if type(e) == None and type(r) == None:
    if e == None and r == None:
        return sqrt(q**2 + w**2)
    else:
        return sqrt((e-q)**2 + (r-w)**2)

#def distance(t, y) -> float:
#    return sqrt(t**2 + y**2)

# --- RADIUS ------------------------------------
"""
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
"""
# updated to query if the two points are a diameter or not, so response can be more accurate
def radius(a, s, d, f) -> float:
    inp = input("Is the line between these two points a diameter? [Y/n]: ")
    # input validation
    while(inp != 'Y' and inp != 'n'):
        inp = input("error: not a valid response [Y/n]: ")
        
    if(inp == 'Y'):
        return distance(distance(a,d)/2,distance(s,f)/2,d,f)
    else:
        return distance(a,s,d,f)

#def radius(g, h) -> float:
#    return distance(g,h)
# Note: this is a remnant from a previous version of the program, when it only used a single
# dimension. 

# --- CIRCUMFERENCE -----------------------------
"""
double circumference(double r = 1)
{
   return 2 * PI * r;
}
"""
def circumference(r) -> float:
    return 2 * PI * r

# --- AREA --------------------------------------
"""
double area(double r = 1)
{
   return PI * pow(r, 2);
}
"""
def area(r) -> float:
    return PI * (r**2)

# --- MAIN ------------------------------------------------------------------------------
# --- INTRODUCTION ------------------------------
"""
int main()
{
   double x1, y1, x2, y2;
   string inp;

   cout << "Points must be entered as two integers, such as: x x" << endl;
"""
# input requirements
print("Points must be entered as two integers, such as: x y")

# --- FIRST POINT -------------------------------
"""
   cout << "Input your first point: ";
   cin >> x1 >> y1;
"""
x1, y1 = map(float, input("Input your first point: ").split())

# --- SECOND POINT ------------------------------
"""
   cout << "Input your second point: ";
   cin >> x2 >> y2;
"""
x2, y2 = map(float, input("Input your second point: ").split())

# --- OPERATION SELECTION -----------------------
"""
   cout << fixed << showpoint << setprecision(3);
   cout << "Possible operations: distance, radius, circumference, area" << endl;
   cout << "What operation would you like to do: ";
   cin >> inp;

   while (inp != "distance" && inp != "radius" && inp != "circumference" && inp != "area") {
      cout << "Not valid: ";
      cin >> inp;
   }
"""
# output decimals, ask for intended operation
print("Possible operations: distance, radius, circumference, area")
inp = input("What operation would you like to do: ")
# input validation
while(inp != "distance" and inp != "radius" and inp != "circumference" and inp != "area"):
    inp = input("error: invalid operation: ")

# --- OPERATION CALLS ---------------------------
"""
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

   return 0;
}
"""
# call functions based on input
if(inp == "distance"):
    print(f"The distance between the two points is {distance(x1,y1,x2,y2):.3f} units.")
elif(inp == "radius"):
    print(f"The radius is {radius(x1,y1,x2,y2):.3f} units.")
elif(inp == "circumference"):
    print(f"The circumference is {circumference(radius(x1,y1,x2,y2)):.3f} units.")
elif(inp == "area"):
    print(f"The area is {area(radius(x1,y1,x2,y2)):.3f} units squared.")
# Note: due to input validation in previous section (OPERATION SELECTION), this else clause should
# not be hit.
else:
    print("whoop de do, I have no clue")
    exit(1)

