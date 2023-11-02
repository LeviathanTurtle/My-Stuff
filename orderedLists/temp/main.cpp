/*

 */

#include <iostream>
#include <fstream>
#include <string>
#include "linkedList.h"
#include "unorderedLinkedList.h"
using namespace std;

int main ()
{
   unorderedLinkedList<string> one;
   unorderedLinkedList<string> two;
   string str;

   ifstream file1;
   cout << "opening file..." << endl;
   file1.open("list1.txt");
   cout << "file opened" << endl;
   cout << "reading in files" << endl;
   while (!file1.eof())
   {
      file1 >> str;
      one.insertLast(str);
      // it's reading in the last element twice I'm not sure why
   }
   file1.close();
   cout << endl;
   one.print();
   cout << endl;


   ifstream file2;
   file2.open("list2.txt");
   while (!file2.eof())
   {
      file2 >> str;
      two.insertLast(str);
   }
   file2.close();
   two.print();
   cout << endl;


   one.merge(two);
   one.print();
   cout << endl;

   return 0;
}

/*
	file.open(str+".txt");
 */
