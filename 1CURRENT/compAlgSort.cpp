/* USER-SPECIFIED SORTING OF RANDOM NUMBERS
 * William Wadsworth + Kolt Byers
 * Created: 11.29.2021
 * Doctored: 10.15.2023
 * CSC 2342
 * ~/discrete/prog2
 * 
 * 
 * This program takes random values from a data file and puts them into an
 * array. The program then calls user-specified sorting algorithms to
 * sort the array from least to greatest. The program outputs the
 * frequency of the swap statements and the number of elements moved.
 * 
 * 
 * To compile:
 *   g++ compAlgSort.cpp -Wall -o compAlgSort
 * 
 * To run:
 *   ./compAlgSort
*/

#define MAX 100000

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
using namespace std;

// =================================================== FUNCTION PROTOTYPES
// bubble sort 1
// all function variable spots are the same,
//              exceptions: loadSort and loadArray

// bubble sort 1
//       array \/
//void bubble1(int[], int, long long int[]);
void bubble1(int*, int, long long int[]);
//            amount ^  results ^

// bubble sort 2
//void bubble2(int[], int, long long int[]);
void bubble2(int*, int, long long int[]);

// mergeSort
//void mergeSort(int[], long long int[], int, int);
void mergeSort(int*, long long int[], int, int);

// merge array (list)
//void merge(int[], long long int[], int, int, int);
void merge(int*, long long int[], int, int, int);

// loadSort
//void loadSort(int[], long long int[], double, int, int);
void loadSort(int*, long long int[], double, int, int);

// loadArray
//void loadArray(ifstream&, int[], int);
void loadArray(ifstream&, int*, int);

// ========================================================== MAIN PROGRAM
int main()
{
   // file io
   ifstream test;
   // data array
   //int data[MAX];

   // query dataset, open file
   string dataResp;
   cout << "Enter the dataset: ";
   cin >> dataResp;
   test.open(dataResp);
   cout << endl;

   // query number of processed items
   int sortAmt = 10; // default 10, ignore faulty input = reduce headaches
   cout << "Enter the number of random numbers to process [0, 100,000]: ";
   cin >> sortAmt;
   // dynamically create data array
   int* data;
   if(typeid(sortAmt) == typeid(int))
       data = new int[sortAmt]; // use user input for size
   else
       data = new int[MAX]; // user input size bad, default to #define

   // load array with data
   loadArray(test, data, sortAmt);

   // query sort method number
   int sortNum = 1; // default 1, more less headaches is good
   cout << endl << "Enter the type of sort (1=bubble1,2=bubble2,3=merge): ";
   cin >> sortNum;
   cout << endl;

   // results, put in array for ease of use
   long long int results[6] = { 0 };
   /*
   results[0] = bubble1 if frequency
          [1] = bubble1 elements moved
          [2] = bubble2 if frequency
          [3] = bubble2 elements moved
          [4] = mergeSort recursive calls
          [5] = mergeSort elements moved
   */

   // ============================================================ TIMEVAL

   struct timeval startTime, stopTime;
   double /*start, stop,*/ diff;

   gettimeofday(&startTime, NULL);
   loadSort(data, results, diff, sortAmt, sortNum);
   gettimeofday(&stopTime, NULL);

/*
   start = startTime.tv_sec + (startTime.tv_usec / 1000000.0);
   stop = stopTime.tv_sec + (startTime.tv_usec / 1000000.0);
   diff = stop - start;
*/    
   diff = stopTime.tv_sec - startTime.tv_sec;
   diff += (stopTime.tv_usec - startTime.tv_usec) / 1000000.0;

   // ====================================================== OUTPUT PART 2
    
   cout << "  time: " << fixed << /*showpoint <<*/ setprecision(10)
        << diff << " seconds " << endl;

   // choice to output sorted array
   string yn = "n"; // default of no output 
   cout << endl << "Output the sorted list (y/n): ";
   cin >> yn;
   if (yn == "y")
      for (int i = 0; i < sortAmt; i++)
         cout << data[i] << " ";
   cout << endl;

   // close file, end program
   test.close();
   return 0;
}

// ================================================== FUNCTION DEFINITIONS

// bubble sort 1 definition
// sorts n values in a[] and increments designated counters in output[]
//void bubble1(int a[], int n, long long int output[])
void bubble1(int* a, int n, long long int output[])
{
   for (int i = 0; i < n-1; i++)
      for (int j = 0; j < n - 1; j++) {
         output[0]++; // if frequency counter
         if (a[j] > a[j + 1]) {
            int temp = a[j];
            output[1]++;
            a[j] = a[j + 1];
            output[1]++;
            a[j + 1] = temp;
            output[1]++; // elements moved  counter
         }
      }
}

// bubble sort 2 definition
// sorts n values in a[] and increments designated counters in output[]
//void bubble2(int a[], int n, long long int output[])
void bubble2(int* a, int n, long long int output[])
{
   for (int i = 0; i < n-1; i++)
      for (int j = 0; j < n - i-1; j++) {
         output[2]++;
         if (a[j] > a[j + 1]) {
            int temp = a[j];
            output[3]++;
            a[j] = a[j + 1];
            output[3]++;
            a[j + 1] = temp;
            output[3]++;
         }
      }
}

// mergeSort sort definition
// splits a[] (via merge method process) in half recursively, increments
//    designated counters in output[], calls merge
//void mergeSort(int a[], long long int output[], int start, int end)
void mergeSort(int* a, long long int output[], int start, int end)
{
   output[4]++;
   if (start < end) {
      int middle = start + (end-start)/2;
      mergeSort(a, output, start, middle);
      mergeSort(a, output, middle + 1, end);
      merge(a, output, start, middle, end);
   }
}

// merge sort definition
// reassembles split arrays in sorted order (least to greatest), increments
//    designated counters in output[]
//void merge(int m[], long long int output[], int front, int midL, int back)
void merge(int* m, long long int output[], int front, int midL, int back)
{
   int T[MAX];
   int L1 = front;
   int R1 = midL;
   int L2 = midL + 1;
   int R2 = back;
   int n = front;
   while (L1 <= R1 && L2 <= R2)
      if (m[L1] < m[L2]) {
         T[n++] = m[L1++];
         output[5]++;
       }
       else {
          T[n++] = m[L2++];
          output[5]++;
       }
   while (L1 <= R1) {
      T[n++] = m[L1++];
      output[5]++;
   }
   while (L2 <= R2) {
      T[n++] = m[L2++];
      output[5]++;
   }
   for (int i = front; i <= back; i++) {
      m[i] = T[i];
      output[5]++;
   }
}

// loadSort definition
// asks user for sorting algorithm, calls designated function, outputs
//    everything except time
//void loadSort(int data[], long long int results[], double time, int sortAmt, int sortNum)
void loadSort(int* data, long long int results[], double time, int sortAmt, int sortNum)
{
   
   // run sort, store selected store frequency in variable
   long int methFreq = 0;
   // same thing with elements moved
   long int elmtMov = 0;

   // call bubble sort 1 if specified
   if (sortNum == 1) {
      bubble1(data, sortAmt, results);
      methFreq = results[0];
      elmtMov = results[1];
   }
   // call bubble sort 2 if specified
   else if (sortNum == 2) {
      bubble2(data, sortAmt, results);
      methFreq = results[2];
      elmtMov = results[3];
   }
   // call merge sort if specified
   else if (sortNum == 3) {
      int start = 0;
      int end = sortAmt-1;
      mergeSort(data, results, start, end);
      // qwuhtiuqwbtiuowqb4utiqbwuigbqw34uigbquio34bgquio34
      methFreq = results[4] - 1;
      elmtMov = results[5];
   }
   else {
      cout << "error: invalid sort method number: ";
      cin >> sortAmt;
   }
   // ^ just in case for some reason it takes something besides 1-4

   if (sortNum == 3) {
      // results output for merge
      cout << "Results:" << endl << endl;
      cout << "  n: " << sortAmt << endl
           << "  sort: " << sortNum << endl
           << "  recursive calls: " << methFreq << endl
           << "  elements moved: " << elmtMov << endl;
   }
   else {
       // results output for bubble
       cout << "Results:" << endl << endl;
       cout << "  n: " << sortAmt << endl
            << "  sort: " << sortNum << endl
            << "  if frequency: " << methFreq << endl
            << "  elements moved: " << elmtMov << endl;
   }
}

// loadArray definition
// load array definition: loads values from data file
//void loadArray(ifstream& dataFile, int array[], int amount)
void loadArray(ifstream& dataFile, int* array, int amount)
{
   for (int i = 0; i < amount; i++)
      dataFile >> array[i];

   cout << "data file successfully loaded" << endl;
}


