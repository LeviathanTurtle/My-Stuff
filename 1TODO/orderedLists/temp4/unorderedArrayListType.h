/*
   Author: William Wadsworth
   Date: 3.22.21
   Class: CSC1720
   Code location: ~/csc1720/prog3/unorderedArrayListType.cpp

   About:
      This is the header file for the unorderedArrayListType class
*/

#ifndef UnorderedArrayList_TYPE
#define UnorderedArrayList_TYPE
   
#include "arrayListType.h" 
using namespace std;

template <class T>
class unorderedArrayListType: public arrayListType<T>
{
   using arrayListType<T>::list;
   using arrayListType<T>::length;
   using arrayListType<T>::maxSize;
   public:
      //Constructor
      unorderedArrayListType(int size = 100);

      /* insertAt Function adds string to list at specified location
       * Precondition: string, location, list are declared and initialized
       * Postcondition: list array will have updated placement and updated length
       */
      void insertAt(int location, T insertItem);
  
      /* insertEnd Function to add item at the end of the list
       * Precondition: insertItem, list array must be declared and initialized
       * Postcondition: final element of the array will be insertItem and the array length will be updated
       */
      void insertEnd(T insertItem);

      /* replaceAt Function replaces value at location with repItem
       * Precondition: location, repItem, array must be declared and initialized
       * Postcondition: list[location] will now store repItem
       */
      void replaceAt(int location, T repItem);
  
      /* seqSearh Function searches array for searchItem
       * Precondition: searchItem, array must be declared and initialized
       * Postcondition: function will return list element address in array if found
       */
      int seqSearch(T searchItem) const;
}; 
#include "unorderedArrayListType.cpp"
#endif
