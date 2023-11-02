/*
   William Wadsworth
   9/16/29
   CSC1710
   ~/csc1710/lab5/lab5.cpp
   lab 5
*/

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
using namespace std;

int main ()
{
   // declare data sheet variables
   ifstream inFile;
   ofstream outFile;

   // initialize variables
   string first, last, department;
   int cups;
   double salary, bonus, taxes, cost, distance, time;
   
   // open data files
   inFile.open("inData.txt");
   outFile.open("outData.txt");

   // take first and last values from data sheet, display in output
   inFile >> first;
   inFile >> last;
   inFile >> department;
   cout << "Name: " << first << " " << last << ", Department: " << department << endl;

   // take salary, bonus, and tax values from data sheet, display in output, set to display 2 decimal values
   cout << fixed << showpoint << setprecision(2);
   inFile >> salary;
   inFile >> bonus;
   inFile >> taxes;
   cout << "Monthly Gross Income: $" << salary << ", Bonus: " << bonus << "%, Taxes: " << taxes << "%" << endl;

   // take distance and time values from data sheet, display in output, calculate mph
   inFile >> distance;
   inFile >> time;
   cout << "Distance traveled: " << distance << " miles, Traveling Time: " << time << " hours" << endl;
   cout << "Average Speed: " << distance/time << " miles per hour" << endl;

   // take cups and cost values from data sheet, display in output
   inFile >> cups;
   inFile >> cost;
   cout << "Number of coffee cups sold: " << cups << ", Cost: $" << cost << " per cup" << endl;
   cout << "Sales amount = $" << cups*cost << endl;

   // read output into the output file
   outFile << inFile;

   // close data files
   inFile.close();
   outFile.close();
   return 0;
}













