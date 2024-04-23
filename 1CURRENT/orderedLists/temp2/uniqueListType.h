#ifndef UniqueArrayList_TYPE
#define UniqueArrayList_TYPE

#include "unorderedArrayListType.h"

class uniqueListType: public unorderedArrayListType
{
   public:
      uniqueListType(int size);

      /* insertAt function adds insertItem at list[location]
	 precondition: location, insertItem, list must be declared and initialized
	 postcondition: list[location] will now store insertItem, list length will be updated
      */
      void insertAt(int location, string insertItem);

      /* insertEnd function adds insertItem at the end of the list
	 precondition: insertItem, list must be declared and initialized
	 postcondition: final element of list will store insertItem, list length will be updated
      */
      void insertEnd(string insertItem);

      /* replcaeAt function replaces list[location] with repItem
	 precondition: location, repItem, list must be declared and initialized
	 postcondition: list[location] will store repItem
      */
      void replaceAt(int location, string repItem);

      /* isCopy function to determine if insertItem is already in the list array
	 precondition: insertItem, list must be declared and initialized
	 postcondition: function will return T/F if insertItem is a copy or not
      */
      bool isCopy(string insertItem);
};

#endif
