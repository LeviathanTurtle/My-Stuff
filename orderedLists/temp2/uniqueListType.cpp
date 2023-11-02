#include <iostream>
#include <string>
#include "uniqueListType.h"

using namespace std;

uniqueListType::uniqueListType(int size)
	      : unorderedArrayListType(size)
{
}

bool uniqueListType::isCopy(string insertItem)
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

void uniqueListType::insertAt(int location, string insertItem)
{
   if (isCopy(insertItem))
      cout << "element already in array" << endl;
   else
      unorderedArrayListType::insertAt(location, insertItem);
}

void uniqueListType::insertEnd(string insertItem)
{
   uniqueListType::insertAt(length-1, insertItem);
}

void uniqueListType::replaceAt(int location, string repItem)
{
   if (isCopy(repItem))
      cout << "element already in array" << endl;
   else
      list[location] = repItem;
}


