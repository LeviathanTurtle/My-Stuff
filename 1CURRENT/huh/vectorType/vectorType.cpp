/*

*/

#include "vectorType.h"
#include <iostream>
#include <iomanip>
#include <cmath>
using namespace std;

/*
   vectorType - default constructor, initialize variables
   pre-condition: x, y, z are declared
   post-condition: x, y, z will be set to parameters
*/
vectorType::vectorType(double vx, double vy, double vz)
{
   x = vx;
   y = vy;
   z = vz;
}

/*
   vectorType - parameterized constructor, initialize variables
   pre-condition: x, y, z are declared
   post-condition: x, y, z will be set to 0
*/
vectorType::vectorType()
{
   x = 0;
   y = 0;
   z = 0;
}

/*
   setCpmps - initialize coords using parameters
   pre-condition: x, y, z must be declared
   post-condition: x, y, z will be initialized
*/
void vectorType::setComps(double sx, double sy, double sz)
{
   x = sx;
   y = sy;
   z = sz;
}

/*
   getX - return private variable x
   pre-condition: x variable is declared and initialized
   post-condition: variable is returned
*/
double vectorType::getX() const
{
   return x;
}

/*
   getY - return private variable y
   pre-condition: y variable is declared and initialized                                                                                                                                 post-condition: variable is returned
*/
double vectorType::getY() const
{
   return y;
}

/*
   getZ - return private variable z
   pre-condition: z variable is declared and initialized
   post-condition: variable is returned 
*/
double vectorType::getZ() const
{
   return z;
}

/*
   printVector - output vector coords
   pre-condition: x, y, z are declared and initialized
   post-condition: none
*/
void vectorType::printVector() const
{
   cout << fixed << showpoint << setprecision(4);
   cout << "<" << x << ", " << y << ", " << z << ">";
}

/*
   calcMagnitude - calculate magnitude of vector
   pre-condition: x, y, z are declared and initialized
   post-condition: magnitude is returned
*/
double vectorType::calcMagnitude(double cx, double cy, double cz)
{
   double sqx = cx*cx, sqy = cy*cy, sqz = cz*cz;
   return sqrt(sqx+sqy+sqz);
}

/*
   scalarMultiply - multiple each coordinate in a vector by a set scale
   pre-condition: x, y, z, scale are declared and initialized
   post-condition: scaled vector is output
*/
void vectorType::scalarMultiply(double mx, double my, double mz, double scale)
{
   cout << fixed << showpoint << setprecision(4);
   cout << "<" << scale*mx << "," << scale*my << "," << scale*mz << ">";
}
