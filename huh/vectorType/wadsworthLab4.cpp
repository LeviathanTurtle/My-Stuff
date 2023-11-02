/*
   Author: William Wadsworth
   Date: 2.2.21
   Class: CSC1720
   Code location: ~/csc1720/lab4/wadsworthLab4.cpp

   About:
   This program will use the vector class to output a 3D coordinate

   The output should be formatted as follows:

      Vector components: (5,8,1)
      Testing setComps method with values (5,8,1)
      Vector components: (0,0,0)
      Testing setComps method with values (0,0,0)

      vectorA:
      <5.0000,8.0000,1.0000>
      vectorB:
      <0.0000,0.0000,0.0000>

      Magnitude: vectorA
      9.4868
      Magnitude: vectorB
      0.0000

      Scaling: vectorA
      <15.0000,24.0000,3.0000>
      Scaling: vectorB
      <0.0000,0.0000,0.0000>

   To compile:
      g++ -Wall wadsworthLab4.cpp -o wadsworthLab4

   To execute:
      ./wadsworthLab4
*/

#include "vectorType.h"
#include "vectorType.cpp"
#include <iostream>
using namespace std;

int main ()
{
   // define vector, give coords for paramaterized constructor
   vectorType vectorA(5,8,1);
   // define vector, blank coords for default constructor
   vectorType vectorB;
   //vectorType vectorC;

   // output vector coords, 
   cout << "Vector components: (" << vectorA.getX() << "," << vectorA.getY() << "," << vectorA.getZ() << ")" << endl;
   cout << "Testing setComps method with values (5,8,1)" << endl;
   vectorA.setComps(vectorA.getX(),vectorA.getY(),vectorA.getZ());   

   cout << "Vector components: (" << vectorB.getX() << "," << vectorB.getY() << "," << vectorB.getZ() << ")" << endl;
   cout << "Testing setComps method with values (0,0,0)" << endl;
   vectorB.setComps(vectorB.getX(),vectorB.getY(),vectorB.getZ());

   cout << endl;
   cout << "vectorA:" << endl;
   vectorA.printVector();
   cout << endl;
   cout << "vectorB:" << endl;
   vectorB.printVector();
   cout << endl << endl;

   cout << "Magnitude: vectorA" << endl;
   cout << vectorA.calcMagnitude(vectorA.getX(),vectorA.getY(),vectorA.getZ()) << endl;
   cout << "Magnitude: vectorB" << endl;
   cout << vectorB.calcMagnitude(vectorB.getX(),vectorB.getY(),vectorB.getZ()) << endl;

   cout << endl;
   cout << "Scaling (3): vectorA" << endl;
   vectorA.scalarMultiply(vectorA.getX(),vectorA.getY(),vectorA.getZ(),3);
   cout << endl;
   cout << "Scaling (3): vectorB" << endl;
   vectorB.scalarMultiply(vectorB.getX(),vectorB.getY(),vectorB.getZ(),3);
   cout << endl;

   return 0;
}

