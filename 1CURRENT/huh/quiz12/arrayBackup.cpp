/*
   William Wadsworth
   CSC1710
   11.5.20
   ~/csc1710/lab11/wadsworthLab11.cpp
   make an array from a selected data file, sort, and find mean, minimum, and maximum
*/

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

// prototypes
void loadArray(double n[], ifstream& file);
void printArray(double numbers[], int index, int size);
void sortArray(double numArray[], int index, int size);
double median(double array[], int size);
double minimum(double array[]);
double maximum(double array[], int size);
int searchArray(double array[], int searchItem, int size);

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
   bool match = false, i = 0;

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

