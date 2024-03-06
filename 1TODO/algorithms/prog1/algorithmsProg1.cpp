/*
   William Wadsworth
   Dr. Williams
   CSC 2710
   2.2.22
 */

#include <iostream>
using namespace std;

// sequential prototype
void sequential(int[], int, int&);

// exchange prototype
void exchange(int[], int);

// binary prototype
int binary(int[], int, int, int, int&);

// recursive Fibonacci prototype
int rFib(int);

// iterative Fibonacci prototype
void iFib(int);

int main()
{
   int array[100];
   // const for array size
   // sizeof ?

   int size = 100;
   cout << "Array size: " << size << endl;


   // int random = (rand() % 100);
   for (int i = 0; i < size; i++)
   {
       array[i] = (rand() % 100);
       cout << array[i] << " ";
   }
   cout << endl;
   

   string input = "binary"; // default of binary, no time for input validation
   cout << "Select an algorithm: ";
   cin >> input;

   int scnt = 0; // counter for sequential
   int bcnt = 0; // counter for binary

   if (input == "sequential")
   {
       sequential(array, size, scnt);

       cout << endl << "items serached: " << scnt << endl;
   }
   else if (input == "exchange")
   {
       exchange(array, size);

       for (int i = 0; i < size; i++)
           cout << array[i] << " ";
       cout << endl << "The list is sorted." << endl;
   }
   else if (input == "binary")
   {
       int resp = 0; // default of 0
       cout << "item to search for: ";
       cin >> resp;
       int start = 0;
       binary(array, start, size, resp, bcnt);

       for (int i = 0; i < size; i++)
           cout << array[i] << " ";
       cout << "The list is sorted." << endl;
   }
   else if (input == "recFib")
   {
       int ans1 = 2, m = 0;
       cout << "how many values: ";
       cin >> ans1;
       if (m < ans1)
       {
           cout << rFib(ans1) << " ";
           m++;
       }
       
   }
   else if (input == "iteFib")
   {
       int ans2 = 2;
       cout << "how many values: ";
       cin >> ans2;
       iFib(ans2);
   }
   else
   {
      cerr << "error: invalid algorithm" << endl;
      return -1;
   }
   
   return 0;
}

// SEQUENTIAL
void sequential(int array[], int n, int& cnt)
{
    bool isFound = false;
    int i = 0;
    while (isFound == false && i < n)
    {
        cnt++;
        if (array[i] == n)
        {
            cnt++;
            isFound = true;
        }
        i++;
    }
}

// EXCHANGE
void exchange(int array[], int n)
{
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (array[j] < array[i])
                swap(array[j], array[i]);
}

// BINARY
int binary(int array[], int start, int end, int item, int& cnt)
{
    if (end >= start)
    {
        cnt++;
        int mid = 1 + (end - 1) / 2;
        if (array[mid] == item)
            return mid;
        if (array[mid] > item)
            return binary(array, start, mid-1, item, cnt);
        cnt++;
        return binary(array, mid + 1, end, item, cnt);
    }
}

// nth FIBONACCI (RECURSIVE)
int rFib(int n)
{
    if (n <= 1 || n == 0)
        return n;
    return rFib(n - 1) + rFib(n - 2);
}

// nth FIBONACCI (ITERATIVE)
void iFib(int n)
{
    int x = 0, y = 1, z = 0;
    for (int i = 0; i < n; i++)
    {
        cout << x << " ";
        z = x + y;
        x = y;
        y = z;
    }
}
