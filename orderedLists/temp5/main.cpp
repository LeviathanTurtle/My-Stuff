/*

 */
#include <iostream>
#include <fstream>
#include "linkedlist.h"
#include "unorderedLinkedList.h"
#include "orderedLinkedList.h"
#include "stuType.h"
using namespace std;

int main ()
{
   ifstream data1, data2;
   orderedLinkedList<stuType> list1, list2;

   cout << "opening first file..." << endl;
   data1.open("list1.txt");
   cout << "file opened." << endl;
   
   data1.close();

   cout << "opening second file..." << endl;
   data2.open("list2.txt");
   cout << "file opened." << endl;

   data2.close();

   return 0;
}

// EPSILON = 0.0001; getGPA();
