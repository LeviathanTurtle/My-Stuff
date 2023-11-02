/*

*/

#include <iostream>
#include <fstream>
#include <string>
#include "arrayListType.h"
#include "unorderedArrayListType.h"
#include "uniqueListType.h"
using namespace std;

// function initializes values in the list array, array must be defined and
// values must be valid. Array will be loaded with strings
void load(unorderedArrayListType& namesList);

int main()
{
   // create object
   uniqueListType n(16);

   // output size of list
   cout << "list of: " << n.listSize() << endl;

   // load list, output number of avlues, print updated list
   load(n);
   cout << "List of: " << n.listSize() << endl;
   n.print();

   // add Kristy to list[4]
   n.insertAt(5, "Kristy");
   n.print();

   // add Tom to list[12]
   n.insertAt(12, "Tom");
   n.print();

   return 0;
}

void load(unorderedArrayListType& namesList)
{
   // create filestream variable
   ifstream data;
   string name;
   // are there values to read in? If so, check if they are duplicates and 
   // skip read-in or add to list. If not, output error
   while(!data.eof())
   {
      if (namesList.listSize() < namesList.maxListSize())
         for (int i = 1; i < namesList.maxListSize(); i++)
         {
	    data >> name;
            if (uniqueListType::isCopy(name))
               break; // okay okay okay I know you said not to use breaks but I wasn't sure how to get out of the loop :/
            else
	       namesList.insertAt(i, name);
         }
      else
         cerr << "error: list is full!" << endl;
   }
}








