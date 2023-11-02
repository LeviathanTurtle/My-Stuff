/* Name: Sample Test Code
   Date: 1-10-2021
   Class: CSC-1720
   Location: ~/csc1720/labs/lab4

   About: This program tests each component of the counterType class.  The 
   expected output is given in the comments, does your class implementation agree?

   How to compile:
   g++ -Wall counterType.cpp lab2Test.cpp -o testProg

   To execute the test program:
   ./testProg
*/
#include <iostream>
#include"counterType.h"

using namespace std;

int main()
{
   //test default constructor
   counterType cntA;

   //test parameter constructor, start counter at 10
   counterType cntB(10);   

   //test parameter constructor, start counter at -10
   counterType cntC(-10);   

   cout << "Constructor and displayCounter Test" << endl;
   cout << "Counter A = ";
   cntA.displayCounter();  // output is a 0
   cout << endl;
   cout << "Counter B = ";
   cntB.displayCounter();  // output is a 10
   cout << endl;
   cout << "Counter C = ";
   cntC.displayCounter();  // output is a 0
   cout << endl;
   cout << endl;

   cout << "setCounter Test with a parameter and then default parameter" << endl;
   cout << "also used getCounter Test to retrieve the updated counter" << endl;
   //sets the counter to 2, ignores the default parameter
   cntA.setCounter(2);     
   cout << "Counter A = " << cntA.getCounter() << endl;

   //NOTE - no parameter, uses the default which is 0
   cntA.setCounter();      
   cout << "Counter A = " << cntA.getCounter() << endl;
   cout << endl;
  
   cout << "3 incrementCounter/ 1 decrementCounter test" << endl; 
   //cntB counter is currently 10
   cout << "Counter B = " << cntB.getCounter() << endl;
   cntB.incrementCounter();
   cntB.incrementCounter();
   cntB.incrementCounter();
   cntB.decrementCounter();

   //with 3 increments and two decrements, 
   //cntB counter should now be 12
   cout << "Counter B = " << cntB.getCounter() << endl;
   cout << endl;

   cout << "1 decrementCounter test on counter at 0" << endl; 
   //cntA counter is currently 0
   cout << "Counter A = " << cntA.getCounter() << endl;
   cntA.decrementCounter();
   cout << "Counter A = " << cntA.getCounter() << endl;
   cout << endl;



   cout << "Test the assignment operator" << endl; 
   cout << "Counter A = " << cntA.getCounter() << endl;
   cout << "Counter B = " << cntB.getCounter() << endl;
   //testing the assignment operator on classes - this is allowed.
   cout << "Assigning B to A" << endl;
   cntA = cntB;  
   //at this point, cntA and cntB should both be equal.
   //however, you cannont compare the two instances of counterType 
   //directly using ==, <=, etc.  You must use the dot operator
   //to compare something from the private areas.
   cout << "Counter A and Counter B are "; 
   if(cntA.getCounter() == cntB.getCounter()) {
      cout << "equal" << endl;
   } else {
      cout << "NOT equal" << endl;
   }
   return 0;
}
