/*
   Author: William Wadsworth
   Date: 3.22.21
   Class: CSC1720
   Code location: ~/csc1720/prog3/uniqueListType.cpp

   About:
      This is the header file for the uniqueListType class
*/

#ifndef UniqueArrayList_TYPE
#define UniqueArrayList_TYPE

#include "unorderedArrayListType.h"
using namespace std;

template <class T>
class uniqueListType: public unorderedArrayListType <T>
{
   using arrayListType<T>::list;
   using arrayListType<T>::length;
   using arrayListType<T>::maxSize;
   public:
      
      uniqueListType(int size);

      /* insertAt function adds insertItem at list[location]
	   * precondition: location, insertItem, list must be declared and initialized
	   * postcondition: list[location] will now store insertItem, list length will be updated
       */
      void insertAt(int location, T insertItem);

      /* insertEnd function adds insertItem at the end of the list
	   * precondition: insertItem, list must be declared and initialized
	   * postcondition: final element of list will store insertItem, list length will be updated
       */
      void insertEnd(T insertItem);

      /* replcaeAt function replaces list[location] with repItem
	   * precondition: location, repItem, list must be declared and initialized
	   * postcondition: list[location] will store repItem
       */
      void replaceAt(int location, T repItem);

      /* isCopy function to determine if insertItem is already in the list array
	   * precondition: insertItem, list must be declared and initialized
	   * postcondition: function will return T/F if insertItem is a copy or not
       */
      bool isCopy(T insertItem);
};
#include "uniqueListType.cpp"
#endif
