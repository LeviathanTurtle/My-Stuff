/*
   Author: William Wadsworth
   Date: 3.25.21
   Class: CSC1720
   Code location: ~/csc1720/lab10/invalidMonth.h

   About:
      This is the header and implementation file for the invalidMonth class
*/

#include <iostream>
using namespace std;

class invalidMonth
{
   private:
      string message;
   public:
      // constructor
      invalidMonth()
      {
         message = "invalid month";
      }
      // parameterized constructor
      invalidMonth(string str)
      {
         message = str;
      }
      /* function huh returns the message from the invalidMonth class
       * precondition: message is declared and initialized
       * postcondition: message is returned to the main program
       */
      string huh()
      {
         return message;
      }
};
