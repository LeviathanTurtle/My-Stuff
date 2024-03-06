/*
   Author: William Wadsworth
   Date: 2.23.21
   Class: CSC1720
   Code location: ~/csc1720/lab7/wadsworthMain.cpp

   About:
   This program tests functions from the arrayListType and unorderedArrayListType classes

   To compile:
      g++ -Wall wadsworthMain.cpp arrayListType.cpp unorderedArrayListType.cpp -o testLab

   To execute:
      ./testLab
*/

#include <iostream>
#include "arrayListType.h"
#include "unorderedArrayListType.h"
using namespace std;

int main()
{
    cout << "ARRAY 1-----" << endl;
    // declare an array x of 10 elements, fill with random numbers
    unorderedArrayListType x(10);
    for (int i = 0; i < x.maxListSize(); i++)
        x.insertEnd(rand() % 100 + 1);

    // test if the function is empty
    cout << "Empty: " << x.isEmpty() << endl;
    // test if the function is full
    cout << "Full: " << x.isFull() << endl;
    // test size of function and loaded values
    cout << "Array is " << x.listSize() << " out of " << x.maxListSize() << endl;

    // print raw array
    x.print();
    cout << endl;

    // remove spot #3 in array, print new array
    x.removeAt(3);
    cout << "Remove item #3: ";
    x.print();
    cout << endl;

    
    cout << "ARRAY 2-----" << endl;
    // declare an array y of 10 elements, fill with random numbers
    unorderedArrayListType y(10);
    for (int i = 0; i < y.maxListSize(); i++)
        y.insertEnd(rand() % 100 + 1);

    // test if the function is empty
    cout << "Empty: " << y.isEmpty() << endl;
    // test if the function is full
    cout << "Full: " << y.isFull() << endl;
    // test size of function and loaded values
    cout << "Array is " << y.listSize() << " out of " << y.maxListSize() << endl;

    // print raw array
    y.print();
    cout << endl;

    // insert int in the first position, print new array
    y.insertFirst(71);
    cout << "Replace first value with 71: ";
    y.print();
    cout << endl;
    
    
    cout << "ARRAY 3-----" << endl;
    // declare an array z of 10 elements, fill with random numbers
    unorderedArrayListType z(10);
    for (int i = 0; i < z.maxListSize(); i++)
        z.insertEnd(rand() % 100 + 1);

    // test if the function is empty
    cout << "Empty: " << z.isEmpty() << endl;
    // test if the function is full
    cout << "Full: " << z.isFull() << endl;
    // test size of function and loaded values
    cout << "Array is " << z.listSize() << " out of " << z.maxListSize() << endl;

    // print raw array
    z.print();
    cout << endl;

    // locate and return max int in the array
    cout << "Max value: " << z.findMax() << endl;
    cout << endl;

    return 0;
}

