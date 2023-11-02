/*
   Author: William Wadsworth
   Date: 3.22.21
   Class: CSC1720
   Code location: ~/csc1720/prog3/arrayListType.h

   About:
      This is the header file for the arrayListType class
*/

#ifndef ArrayList_TYPE
#define ArrayList_TYPE 

#include <iostream>
#include <string>
using namespace std; 

template <class T>
class arrayListType 
{
   protected:
      T *list;    //array to hold the list elements
      int length;   //variable to store the length of the list
      int maxSize;  /* variable to store the maximum 
                     * size of the list
                     */
   public:
      /* Constructor
       * Creates an array of the size specified by the parameter size, default size is 100
       * Precondition: 
       * Postcondition: The list points to the array, length = 0, and maxSize = size;
       */
      arrayListType(int size = 100);

      // Copy constructor
      arrayListType(const arrayListType<T>& otherList);

      // Destructor, deallocates the memory occupied by the array.
      virtual ~arrayListType();

      /* isEmpty Function to determine whether the list is empty
       * Precondition: 
       * Postcondition: Returns true if the list is empty; otherwise, returns false.
       */
      bool isEmpty() const;

      /* isFull Function to determine whether the list is full
       * Precondition: 
       * Postcondition: Returns true if the list is full; otherwise, returns false.
       */
      bool isFull() const;

      /* listSize Function to determine the number of elements in the list.
       * Precondition: 
       * Postcondition: Returns the value of length.
       */
      int listSize() const;

      /* maxListSize Function to determine the maximum size of the list
       * Precondition: 
       * Postcondition: Returns the value of maxSize.
       */
      int maxListSize() const;

      /* print Function to output the elements of the list
       * Precondition: 
       * Postcondition: Elements of the list are output on the standard output device.
       */
      //ostream& print() const;
      void print() const;

      /* insertAt Function to insert insertItem in the list at the position specified by location. 
       * Precondition: 
         Postcondition: Starting at location, the elements of the list are shifted down, 
                        list[location] = insertItem; length++; If the list is full or location is
                        out of range, an appropriate message is displayed.
      */
      virtual void insertAt(int location, string insertItem) = 0;

      /*insertEnd Function to insert insertItem an item at the end of the list
       * Precondition: 
       * Postcondition: list[length] = insertItem; and length++; If the list is full, an appropriate 
       *                message is displayed.
       */
      virtual void insertEnd(string insertItem) = 0;

      /* removeAt Function to remove the item from the list at the position specified by location 
       * Precondition: 
       * Postcondition: The list element at list[location] is removed and length is decremented by 1.
       *                If location is out of range, an appropriate message is displayed.
       */
      void removeAt(int location);

      /* clearList Function to remove all the elements from the list. After this operation, the 
       *           size of the list is zero.
       * Precondition: 
       * Postcondition: length = 0;
       */
      void clearList();

      /* seqSearch Function to search the list for searchItem.
       * Precondition: 
       * Postcondition: If the item is found, returns the location in the array where the item   
       *                is found; otherwise, returns -1.
       */
      virtual int seqSearch(string searchItem) const = 0;

      T getAt(int pos);
};

#include "arrayListType.cpp"
#endif
