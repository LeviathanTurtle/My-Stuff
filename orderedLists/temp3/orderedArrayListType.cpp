/*
   Author: William Wadsworth
   Date: 3.17.21
   Class: CSC1720
   Code location: ~/csc1720/lab9/orderedArrayListType.cpp

   About:
   This is the implementation file for the orderedArrayListType class
*/

#include <iostream>
#include "orderedArrayListType.h" 

using namespace std; 

orderedArrayListType::orderedArrayListType(int size)
                    : arrayListType(size)
{
}  //end constructor

void orderedArrayListType::insertAt(int location, int insertItem) 
{
   if (location < 0 || location >= maxSize || location > length)
      cout << "The position of the item to be inserted "
           << "is out of range." << endl;
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

void orderedArrayListType::insertEnd(int insertItem)
{
   if (length >= maxSize)  //the list is full
      cout << "Cannot insert in a full list." << endl;
   else
   {
      list[length] = insertItem; //insert the item at the end
      length++; //increment the length
   }
} //end insertEnd


void orderedArrayListType::insert(int insertItem)
{
   // input validation
   if (length == maxSize)
      cout << "error: list is full" << endl;
   else
   {
      int spot = 0;
      bool there = false;
      // find where it fits
      while (!there && spot < length)
      {
	    if (list[spot] > insertItem)
	       there = true;
	    else
	       spot++;
      }

      length++;
      // insert
      for (int i = length; i > spot; i--)
	  list[i] = list[i-1];

      list[spot] = insertItem;
   }
} // end insert


void orderedArrayListType::replaceAt(int location, int repItem)
{
   if (location < 0 || location >= length)
      cout << "The location of the item to be "
           << "replaced is out of range." << endl;
   else
      list[location] = repItem;
} //end replaceAt

int orderedArrayListType::seqSearch(int searchItem) const
{
   int loc;
   bool found = false;

   loc = 0;

   while (loc < length && !found)
      if (list[loc] >= searchItem)
         found = true;
      else
         loc++;

   if (found)
      return loc;
   else
      return -1;
} //end seqSearch



const orderedArrayListType& orderedArrayListType::operator=(const orderedArrayListType& rightObject)
{
    orderedArrayListType *ptr = new orderedArrayListType;
    if (ptr != &rightObject)
    {
        ptr->maxSize = rightObject.maxSize;
        ptr->length = rightObject.length;

        ptr->list = new int(maxSize);

        for (int j = 0; j < length; j++)
            ptr->list[j] = rightObject.list[j];
    }

    return *ptr;
}// end operator= overload

ostream& operator<<(ostream& out, orderedArrayListType array)
{
    for (int i = 1; i < array.maxListSize(); i++)
        out << array.list[i-1] << " ";
    return out;
}// end operator<< overload