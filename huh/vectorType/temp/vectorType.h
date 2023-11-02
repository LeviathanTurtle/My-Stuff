/*

   Author: William Wadsworth
   Date: 2.11.21
   Class: CSC1720
   Code location: ~/csc1720/prog2/vectorType.cpp

   About:
   This file contains the vectorType class definition
*/

#ifndef VECTOR_TYPE
#define VECTOR_TYPE

class vectorType
{
   private:
      double x;
      double y;
      double z;
   public:
      vectorType(double, double, double);
      vectorType();
      void setComps(double, double, double);
      double getX() const;
      double getY() const;
      double getZ() const;
      void printVector() const;
      double calcMagnitude();
      vectorType scalarMultiply(double);
      vectorType addVector(vectorType);
      vectorType subVector(vectorType);
      vectorType unitVector(vectorType);
      double dotProduct(vectorType, vectorType);
      vectorType crossProduct(vectorType, vectorType);
      bool equalVector(vectorType, vectorType);
};

#endif

