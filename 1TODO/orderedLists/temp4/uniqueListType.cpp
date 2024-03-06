/*
   Author: William Wadsworth
   Date: 3.22.21
   Class: CSC1720
   Code location: ~/csc1720/prog3/uniqueListType.cpp

   About:
      This is the implementation file for the uniqueListType class
*/

#include <iostream>
#include <string>
//#include "uniqueListType.h"
using namespace std;

template <class T>
uniqueListType<T>::uniqueListType(int size)
	          : unorderedArrayListType<T>(size)
{
}

template <class T>
bool uniqueListType<T>::isCopy(T insertItem)
{
   for (int i = 0; i < length; i++)
   {
      if (list[i] == insertItem)
	     return false;
      else
	     return true;
   }
   return false;
}

template <class T>
void uniqueListType<T>::insertAt(int location, T insertItem)
{
   if (isCopy(insertItem))
      cout << "element already in array" << endl;
   else
      unorderedArrayListType<T>::insertAt(location, insertItem);
}

template <class T>
void uniqueListType<T>::insertEnd(T insertItem)
{
   uniqueListType::insertAt(length-1, insertItem);
}

template <class T>
void uniqueListType<T>::replaceAt(int location, T repItem)
{
   if (isCopy(repItem))
      cout << "element already in array" << endl;
   else
      list[location] = repItem;
}


