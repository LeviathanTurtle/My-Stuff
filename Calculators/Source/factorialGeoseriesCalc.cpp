/* FACTORIAL AND GEOSERIES CALCULATOR
 * William Wadsworth
 * CSC1710
 * Created: 10.15.2020
 * Doctored: 10.15.2023
 * ~/csc1710/lab9/wadsworthlab9.cpp
 * 
 * The binary was last compiled on 5.24.2024.
 * 
 * Usage:
 * To compile: g++ factorialGeoseriesCalc.cpp -Wall -o <exe name>
 * To run: ./<exe name>
*/

#include <iostream>
#include <sstream>
using namespace std;

// function prototpyes
long int factorial(int);
long int dfactorial(int);
double geoseries(double, int, double);

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
   // check inp
   while(inp != "factorial" && inp != "dfactorial" && inp != "geometric") {
      cout << "Please enter 'factorial', 'dfactorial', or 'geometric': ";
      cin >> inp;
   }


   // if the user wants factorial calculation
   if (inp == "factorial") {
      // parameters for input, store in variable
      cout << "Integer must be between 0 and 1,000" << endl << " " << endl;
      cout << "Enter an integer to the nearest whole for factorial calculation: ";
      cin >> endPoint;

      // input validation
      while (endPoint > 1000 || endPoint < 0) {
         cout << "Not valid, integer must be between 0 and 1,000: ";
         cin >> endPoint;
      }

      // output calculation
      cout << endPoint << "! = " << factorial(endPoint) << endl;
   }



   // if the user wants double factorial calculation
   else if (inp == "dfactorial") {
      cout << "Integer must be between 0 and 1000" << endl << " " << endl;
      cout << "Enter an integer to the nearest whole for double factorial calculation: ";
      cin >> endPoint;

      // input validation, integer must be odd
      while (endPoint % 2 == 0 || endPoint > 1000 || endPoint < 0) {
         cout << "Not valid, integer must be odd and between 0 and 1,000: ";
         cin >> endPoint;
      }

      // output calculation
      cout << endPoint << "!! = " << dfactorial(endPoint) << endl;
   }



   else {
      // declare variables
      double a, r = 0.5, t;
      //double sum = 0;
      //int count;

      // title, ask for input
      cout << "Sum of Geometric series" << endl << " " << endl;
      cout << "What is your first term: ";
      cin >> a;

      cout << "How many terms would you like to take the sum of: ";
      cin >> t;

      cout << "Enter your common ratio of choice (enter -1 for 0.5): ";
      cin >> r;
      if(r == -1)
         r = 0.5;

      // display output
      cout << "Sum of " << t << " terms, a = " << a << ", r = " << r << ", is " << geoseries(a, t, r) << endl;
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
double geoseries (double a, int t, double r/*=0.5*/)
{
   double sum = 0;

   for (int i=0; i < t; i++) {
      sum += a;
      a *= r;
   }

   return sum;
} 






















