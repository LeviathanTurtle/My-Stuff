/*
   Author: William Wadsworth
   Date: 3.22.21
   Class: CSC1720
   Code location: ~/csc1720/prog3/unorderedArrayListType.cpp

   About:
      This is the implementation file for the unorderedArrayListType class
*/

#include <iostream>
#include <string>
#include <fstream>
//#include "unorderedArrayListType.h"
using namespace std; 

template <class T>
unorderedArrayListType<T>::unorderedArrayListType(int size)
                         : arrayListType<T>(size)
{
}  //end constructor

template <class T>
void unorderedArrayListType<T>::insertAt(int location, T insertItem) 
{
   if (location < 0 || location >= maxSize || location > length)
      cout << "The position of the item to be inserted is out of range." << endl;
   else if (length >= maxSize)  //list is full
      cout << "Cannot insert in a full list" << endl;
   else
   {
      for (int i = length; i > location; i--)
          list[i] = list[i - 1];	//move the elements down

      list[location] = insertItem; //insert the item at 
                                   //the specified position

      length++;	//increment the length
   }
} //end insertAt

template <class T>
void unorderedArrayListType<T>::insertEnd(T insertItem)
{
   if (length >= maxSize)  //the list is full
      cout << "Cannot insert in a full list." << endl;
   else
   {
      list[length] = insertItem; //insert the item at the end
      length++; //increment the length
   }
} //end insertEnd

template <class T>
void unorderedArrayListType<T>::replaceAt(int location, T repItem)
{
   if (location < 0 || location >= length)
      cout << "The location of the item to be replaced is out of range." << endl;
   else
       list[location] = repItem;
} //end replaceAt

template <class T>
int unorderedArrayListType<T>::seqSearch(T searchItem) const
{
   int loc = 0;
   bool found = false;

   while (loc < length && !found)
      if (list[loc] == searchItem)
         found = true;
      else
         loc++;

   if (found)
      return loc;
   else
      return -1;
} //end seqSearch

















