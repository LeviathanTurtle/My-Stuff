#ifndef UnorderedArrayList_TYPE
#define UnorderedArrayList_TYPE
   
#include "arrayListType.h" 

class unorderedArrayListType: public arrayListType
{
   public:
      //Constructor
      unorderedArrayListType(int size = 100);

      /* insertAt Function adds string to list at specified location
         Precondition: string, location, list are declared and initialized
         Postcondition: list array will have updated placement and updated length
      */
      void insertAt(int location, string insertItem);
  
      /* insertEnd Function to add item at the end of the list
         Precondition: insertItem, list array must be declared and initialized
         Postcondition: final element of the array will be insertItem and the array length will be updated
      */
      void insertEnd(string insertItem);

      /* replaceAt Function replaces value at location with repItem
         Precondition: location, repItem, array must be declared and initialized
         Postcondition: list[location] will now store repItem
      */
      void replaceAt(int location, string repItem);
  
      /* seqSearh Function searches array for searchItem
         Precondition: searchItem, array must be declared and initialized
         Postcondition: function will return list element address in array if found
      */
      int seqSearch(string searchItem) const;

      /* load function initializes array with strings 
         precondition: namesList, list must be declared and initialized
	 postcondition: list array will be loaded with namesList strings
      */
      void load(unorderedArrayListType& namesList) const;
}; 

#endif
