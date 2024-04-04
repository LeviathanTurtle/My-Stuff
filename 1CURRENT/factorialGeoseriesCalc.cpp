/* FACTORIAL AND GEOSERIES CALCULATOR
   William Wadsworth
   CSC1710
   Created: 10.15.2020
   Doctored: 10.15.2023
   ~/csc1710/lab9/wadsworthlab9.cpp
   Factorial calculations + functions
*/

#include <iostream>
using namespace std;

// function prototpyes
long int factorial(int);
long int dfactorial(int);
double geoseries(double, int);

int main () 
{
   // declare variables
   long int endPoint;
   //int x, y = 1, prod = 1;


   // title
   cout << "Factorial and Geometric series calculations" << endl << " " << endl;

   // ask user preference
   string inp;
   cout << "Enter 'factorial', 'dfactorial', or 'geometric': ";
   cin >> inp;



   // if the user wants factorial calculation
   if (inp == "factorial") {
      // parameters for input, store in variable
      cout << "Integer must be between -1,000 and 1,000" << endl << " " << endl;
      cout << "Enter an integer to the nearest whole for factorial calculation: ";
      cin >> endPoint;

      // input validation
      while (endPoint >= 1000 || endPoint <= -1000) {
         cout << "Not valid, integer must be between -1,000 and 1,000: ";
         cin >> endPoint;
      }

      // output calculation
      cout << endPoint << "! = " << factorial(endPoint) << endl;
   }



   // if the user wants double factorial calculation
   else if (inp == "dfactorial") {
      cout << "Integer must be between -1000 and 1000" << endl << " " << endl;
      cout << "Enter an integer to the nearest whole for double factorial calculation: ";
      cin >> endPoint;

      // input validation, integer must be odd
      while (endPoint % 2 == 0 || endPoint >= 1000 || endPoint <= -1000) {
         cout << "Not valid, integer must be odd and between -1,000 and 1,000: ";
         cin >> endPoint;
      }

      // output calculation
      cout << endPoint << "!! = " << dfactorial(endPoint) << endl;
   }



   else {
      // declare variables
      double a, r = 0.5, t, sum = 0;
      int count;

      // title, ask for input
      cout << "Sum of Geometric series" << endl << " " << endl;
      cout << "What is your first term: ";
      cin >> a;

      cout << "How many terms would you like to take the sum of: ";
      cin >> t;

      // display output
      cout << "Sum of " << t << " terms, a = " << a << ", r = 0.5, is " << geoseries(a, t) << endl;
   }

   return 0;
}

// factorial function
long int factorial (int endPoint)
{
   //long int epoint;
   long int prod = 1;

   for (int x=1; x <= endPoint; x++) {
      prod *= x;
      //cout << prod << " ";
   }

   return prod;
}

// double factorial function
long int dfactorial (int endPoint)
{
   //long int epoint;
   int x = 1, prod = 1;

   do
   {
      prod *= x;
      x += 2;
   }
   while (x <= endPoint);

   return prod;
}

// geometric series function
double geoseries (double a, int t)
{
   double sum = 0;

   for (int count=0; count < t; count++) {
      sum += a;
      a *= .5;
   }

   return sum;
} 






















