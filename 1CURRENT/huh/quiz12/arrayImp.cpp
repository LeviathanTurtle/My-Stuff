#include <iostream>
#include <fstream>
#include <string>
using namespace std;

// load array
void loadArray(double n[], ifstream& file)
{
   int index = 0;

   file >> n[index];
   while (n[index] != -1)
   {
      index++;
	  file >> n[index];
   }
}

// print array
void printArray(double numbers[], int index, int size)
{
   while (index < size)
   {
      cout << "nums[" << index << "] = " << numbers[index] << endl;
         index++;
   }
}

// sort array
void sortArray(double numArray[], int index, int size)
{
   int i = 0;
   int hold, pass;

   for (pass = 1; pass < size; pass++)
   {
      for (i = 0; i < size - pass; i++)
      {
         if (numArray[i] > numArray[i + 1])
         {
            hold = numArray[i];
	    numArray[i] = numArray[i + 1];
	    numArray[i + 1] = hold;
	 }
      }
   }
}

// find median in array
double median(double array[], int size)
{
   double med;

   if (size % 2 != 0)
      med = array[size / 2];
   else
      med = (array[size / 2]) + (array[(size / 2) - 1]) / 2;

   return med;
}

// find minimum in array
double minimum(double array[])
{
   return array[0];
}

// find maximum in array
double maximum(double array[], int size)
{
   return array[size - 1];
}

// search array for value
int searchArray(double array[], int searchItem, int size)
{
   bool match = false;
   int i = 0;

   while (i < size)
   {
      if (array[i] == searchItem)
      {
         match = true;
         return match;
      }
      else
         i++;
   }

   return match;
}

