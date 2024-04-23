/*
   Author: William Wadsworth
   Date: 3.17.21
   Class: CSC1720
   Code location: ~/csc1720/lab9/wadsworthMain.cpp

   About:
   This program tests functions from the arrayListType and orderedArrayListType classes

   To compile:
      g++ -Wall wadsworthMain.cpp arrayListType.h arrayListType.cpp orderedArrayListType.h orderedArrayListType.cpp -o testLab

   To execute:
      ./testLab
*/

#include <iostream>
#include "arrayListType.h"
#include "orderedArrayListType.h"
using namespace std;

int main()
{
    orderedArrayListType x(7);
    // fill x list array with random values (1-100), print
    cout << "list x: " << x.maxListSize() << " elements" << endl;
    for (int i = 1; i < x.maxListSize(); i++)
        x.insert(rand() % 100 + 1);
    x.print();
    cout << endl;
    
    // deep copy construct y
    orderedArrayListType y = x;

    cout << "list y: " << y.maxListSize() << " elements" << endl;
    y.print();
    cout << endl;
    
    cout << "testing replaceAt function for deep copy constructor success" << endl;
    cout << "replace 0 for element 4 in array x:" << endl;
    x.replaceAt(3, 0);
    x.print();


    return 0;
}

