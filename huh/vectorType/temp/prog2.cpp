/*
   Author: William Wadsworth
   Date: 2.11.21
   Class: CSC1720
   Code location: ~/csc1720/prog2/prog2WadsworthA.cpp

   About:
   This (A-level, 6 function) program will use the vector class from lab4 to
      add, subtract, compute a unit vector, dot and cross product, and a
      comparison method, in addition to the methods from lab4: a default and
      parameterized constructor, set/get methods, print, scale, and magnitude
       methods.

   To compile:
      g++ -Wall prog2WadsworthA.cpp vectorType.cpp -o theyAreNotPajamas 

   To execute:
      ./theyAreNotPajamas
*/

#include <iostream>
#include "vectorType.h"
// why can't we use this? 
//#include "vectorType.cpp"

using namespace std;

#define EPSILON 0.0001

int main()
{
   // define vector, give coords for paramaterized constructor
   vectorType vectorA(97, 81, 11);
   // define vector, blank coords for default constructor
   vectorType vectorB;


   // VECTOR A

   cout << "------------VECTOR A:" << endl << endl;

   // output components, organize into notation
   cout << "Components: (" << vectorA.getX() << "," << vectorA.getY() << ","
	    << vectorA.getZ() << "): ";
   vectorA.printVector();
   //test setComps method with values (97,81,11)
   vectorA.setComps(vectorA.getX(), vectorA.getY(), vectorA.getZ());

   cout << endl;

   //==========================================================================
   // MAGNITUDE
   // calculate magnitude, output
   cout << "Magnitude: " << vectorA.calcMagnitude() << endl;
   
   //==========================================================================
   // SCALING
   // define scaled vector, output
   vectorType scaledA = vectorA.scalarMultiply(3);
   cout << "Scaling (3): ";
   scaledA.printVector();

   //==========================================================================
   // ADD
   cout << endl << endl << "Addition: " << endl;

   // output base vectorA
   cout << "vectorA: ";
   vectorA.printVector();
   cout << endl;

   // define new vector to add to vectorA, output
   vectorType vectorC(46, 98, 2);
   cout << "vectorC: ";
   vectorC.printVector();
   cout << endl;

   // define new vector to store sum of vectorA + vectorC, output
   vectorType AplusC = vectorA.addVector(vectorC);
   cout << "Sum:     ";
   AplusC.printVector();
   cout << endl;

   //==========================================================================
   // SUBTRACT
   cout << endl << "Subtraction: " << endl;
   
   // output base vectorA 
   cout << "vectorA:    ";
   vectorA.printVector();
   cout << endl;

   // output base vectorC
   cout << "vectorC:    ";
   vectorC.printVector();
   cout << endl;
   
   // define new vector to store difference of vectorA - vectorC, output
   vectorType AminusC = vectorA.subVector(vectorC);
   cout << "Difference: ";
   AminusC.printVector();
   cout << endl;

   //==========================================================================
   // UNIT

   cout << endl << "Unit vector for A: ";

   // define new vector to store unit vector
   vectorType vectorUnitA = vectorA.unitVector(vectorA);
   vectorUnitA.printVector();
   cout << endl;
   
   //==========================================================================
   // DOT PRODUCT
   // output dot product of vectorA and vectorC
   cout << endl << "Dot product for vectorA and vectorC: " 
	<< vectorA.dotProduct(vectorA, vectorC) << endl << endl;
   
   //==========================================================================
   // CROSS PRODUCT
   // define new vector to store cross product of vectorB and vectorD, output
   vectorType crossA = vectorA.crossProduct(vectorA, vectorC);
   cout << "Cross product: ";
   crossA.printVector();
   cout << endl;


   
   //==========================================================================
   //==========================================================================
   //==========================================================================
   cout << endl << endl << endl;

   
   // VECTOR B

   cout << "------------VECTOR B:" << endl << endl;

   // output components, organize into notation
   cout << "Components: (" << vectorB.getX() << "," << vectorB.getY() << ","
	    << vectorB.getZ() << "): ";
   vectorB.printVector();
   //test setComps method with default values
   vectorB.setComps(vectorB.getX(), vectorB.getY(), vectorB.getZ());

   cout << endl;

   //==========================================================================
   // MAGNITUDE
   // calculate magnitude, output
   cout << "Magnitude: " << vectorB.calcMagnitude() << endl;
   
   //==========================================================================
   // SCALING
   // define scaled vector, output
   vectorType scaledB = vectorB.scalarMultiply(3);
   cout << "Scaling (3): ";
   scaledB.printVector();

   //==========================================================================
   // ADD
   cout << endl << endl << "Addition: " << endl;

   // output base vectorB
   cout << "vectorB: ";
   vectorB.printVector();
   cout << endl;

   // define new vector to add to vectorB, output
   vectorType vectorD(72, 41, 28);
   cout << "vectorD: ";
   vectorD.printVector();
   cout << endl;

   // define new vector to store sum of vectorB + vectorD, output
   vectorType BplusD = vectorB.addVector(vectorD);
   cout << "Sum:     ";
   BplusD.printVector();
   cout << endl;

   //==========================================================================
   // SUBTRACT
   cout << endl << "Subtraction: " << endl;

   // output base vectorB
   cout << "vectorB:    ";
   vectorB.printVector();
   cout << endl;

   // output base vectorD
   cout << "vectorD:    ";
   vectorD.printVector();
   cout << endl;

   // define new vector to store difference of vectorB - vectorD, output
   vectorType BminusD = vectorB.subVector(vectorD);
   cout << "Difference: ";
   BminusD.printVector();
   cout << endl;

   //==========================================================================
   // UNIT
   /* vectorD works but for consistency I'm using vectorB also it looks 
    * different and I like that
    */
   cout << endl << "Unit vector for B: ";

   // define new vector to store unit vector
   vectorType vectorUnitB = vectorB.unitVector(vectorB);
   vectorUnitB.printVector();
   cout << endl << endl;

   //==========================================================================
   // DOT PRODUCT
   // output dot product of vectorB and vectorD
   cout << "Dot product for vectorB and vectorD: " 
	<< vectorB.dotProduct(vectorB, vectorD) << endl << endl;

   //==========================================================================
   // CROSS PRODUCT
   // define new vector to store cross product of vectorB and vectorD, output
   vectorType crossB = vectorB.crossProduct(vectorB, vectorD);
   cout << "Cross product: ";
   crossB.printVector();
   cout << endl << endl;
   
   //==========================================================================
   // COMPARISON
   cout << "------------Comparisonm:" << endl << endl;

   // are the coordinates in the vectors equal?
   if (vectorA.equalVector(vectorA, vectorB))
      cout << "Vectors A and B are equal" << endl;
   else
      cout << "Vectors A and B are not equal to within " << EPSILON << endl;

   return 0;
}

/*
 * I don't think we need "vectorType::" for the functions in the implementation,
 * but I figured it would be good practice for inheritance. It does make the main
 * program look weird though (like in unitVector and crossProduct), as I'm sure
 * you've noticed. Also, is it possible to upload files to Spock instead of 
 * having to copy/paste?
*/
