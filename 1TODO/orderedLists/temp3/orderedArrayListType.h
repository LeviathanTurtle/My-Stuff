/*
   Author: William Wadsworth
   Date: 3.17.21
   Class: CSC1720
   Code location: ~/csc1720/lab9/orderedArrayListType.h

   About:
   This is the header file for the orderedArrayListType class
*/

#ifndef UnorderedArrayList_TYPE
#define UnorderedArrayList_TYPE
   
#include "arrayListType.h" 
using namespace std;

class orderedArrayListType: public arrayListType
{
   public:
      //Constructor
      orderedArrayListType(int size = 100);

      /* insertAt Function inserts item into list
         Precondition: location, insertItem, list are declared and initialized
         Postcondition: location in list will be insertItem
      */
      void insertAt(int location, int insertItem);
  
      /* insertEnd Function appends insertItem at the end of list
         Precondition: insertItem, list are declared and initialized
         Postcondition: last element of list will be insertItem
      */
      void insertEnd(int insertItem);

      /* replaceAt Function replaces list[location] with repItem
         Precondition: location, repItem, list are declared and initialized
         Postcondition: list[location] will be repItem
      */
      void replaceAt(int location, int repItem);
  
      /* seqSearh Function searches list for searchItem
         Precondition: searchItem, list are declared and initialized
         Postcondition: if found, location of searchItem in list is returned
      */
      int seqSearch(int searchItem) const;

      /* seqSearh Function inserts insertItem in ordered place in list
         Precondition: insertItem, list are declared and initialized
         Postcondition: insertItem will be in its ordered place in list
      */
      void insert(int insertItem);


      const orderedArrayListType& operator=(const orderedArrayListType&);

      friend ostream& operator<<(ostream&, orderedArrayListType);
}; 

#endif
