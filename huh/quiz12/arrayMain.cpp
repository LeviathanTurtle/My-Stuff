#include "array.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main()
{
   int i = 0;
   string inp;

   // arrays
   double nums[10];
   string options[5] = { "data", "data1", "data2", "data3", "data4" };

   // prompt file
   cout << "Select a data file to open (data, data1, data2, data3, data4): ";
   cin >> inp;

   // data variable, open user-specified file
   ifstream dataFile;
   if (inp == "data")
      dataFile.open("data.txt");
   else if (inp == "data1")
      dataFile.open("data1.txt");
   else if (inp == "data2")
      dataFile.open("data2.txt");
   else if (inp == "data3")
      dataFile.open("data3.txt");
   else if (inp == "data4")
      dataFile.open("data4.txt");
   else
   {
      cout << "Not valid: ";
      cin >> inp;
   }

   // read in values from file, count how many total values
   loadArray(nums, dataFile);

   // define size, set to 0 so we don't include -1
   int size = 0;
    
   // calculate size
   for (int x = 0; nums[x] != -1; x++)
      size++;

   // PRE SORT
   // reset index variable
   i = 0;
   cout << "PRE-SORT ARRAY (RAW)" << endl;
   printArray(nums, i, size);

   // SORT
   // reset index variable
   i = 0;
   sortArray(nums, i, size);

   // POST SORT
   // reset index variable
   i = 0;
   cout << endl << "POST-SORT ARRAY (EDITED)" << endl;
   printArray(nums, i, size);

   // SIZE, MEDIAN, MINIMUM, MAXIMUM
   cout << endl;
   cout << "Size: " << size << " values" << endl << endl;
   cout << "Median: " << median(nums, size) << endl;
   cout << "Minimum: " << minimum(nums) << endl;
   cout << "Maximum: " << maximum(nums, size) << endl;

   int numSearch;
   cout << endl << "Enter number: ";
   cin >> numSearch;

   // show how many recurrences
   cout << "Number of occurrences of the integer " << numSearch << ": " << searchArray(nums, numSearch, size) << endl;

   // close data file, end program
   dataFile.close();
   return 0;
}

