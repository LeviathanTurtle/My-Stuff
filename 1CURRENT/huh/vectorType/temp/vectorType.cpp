/*
   Author: William Wadsworth
   Date: 2.11.21
   Class: CSC1720
   Code location: ~/csc1720/prog2/vectorType.cpp

   About:
   This file contains the implementation of the updated vectorType class
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
double vectorType::calcMagnitude()
{
   double sqx = x*x, sqy = y*y, sqz = z*z;
   return sqrt(sqx+sqy+sqz);
}

/*
   scalarMultiply - multiple each coordinate in a vector by a set scale
   pre-condition: x, y, z, scale are declared and initialized
   post-condition: scaled vector is output
*/
vectorType vectorType::scalarMultiply(double scale)
{
   //cout << fixed << showpoint << setprecision(4);
   vectorType scaledv(scale*x, scale*y, scale*z);
   return scaledv;
}

/*
   addVector - add two vectors together
   pre-condition: x, y, z, add are declared and initialized
   post-condition: new vector S is returned to main
*/
vectorType vectorType::addVector(vectorType add)
{
   vectorType S(x+add.getX(), y+add.getY(), z+add.getZ());
   return S;
}

/*
   subVector - subtract two vectors 
   pre-condition: x, y, z, sub are declared and initialized
   post-condition: new vector is returned to main
*/
vectorType vectorType::subVector(vectorType sub)
{
   vectorType D(x-sub.getX(), y-sub.getY(), z-sub.getZ());
   return D;
}

/*
   unitVector - calculate the unit vector 
   pre-condition: x, y, z, v are declared and initialized
   post-condition: new vector u is returned to main
*/
vectorType vectorType::unitVector(vectorType v)
{
   vectorType u(v.getX()/v.calcMagnitude(),v.getY()/v.calcMagnitude(),v.getZ()/v.calcMagnitude());
   return u;
}

/*
   dotProduct - compute the dot product of two vectors
   pre-condition: x, y, z, d1, d2 are declared and initialized
   post-condition: scalar number is returned to main
*/
double vectorType::dotProduct(vectorType d1, vectorType d2)
{
   return d1.getX()*d2.getX()+d1.getY()*d2.getY()+d1.getZ()*d2.getZ();
}

/*
   crossProduct - calculate the perpendicular vector to c1 and c2
   pre-condition: x, y, z, c1, c2 are declared and initialized
   post-condition: new vector cProd is returned to main
*/
vectorType vectorType::crossProduct(vectorType c1, vectorType c2)
{
   vectorType cProd((c1.getY()*c2.getZ()-c1.getZ()*c2.getY()),(c1.getZ()*c2.getX()-c1.getX()*c2.getZ()),(c1.getX()*c2.getY()-c1.getY()*c2.getX()));
   return cProd;
}

/*
   equalVector - compares the coordinates in two vectors to determine if they are equal
   pre-condition: one, two are declared and initialized
   post-condition: T/F is returned to main
*/
bool vectorType::equalVector(vectorType one, vectorType two)
{
   if (one.getX()==two.getX() && one.getY()==two.getY() && one.getZ()==two.getZ())
      return true;
   else
      return false;
}
