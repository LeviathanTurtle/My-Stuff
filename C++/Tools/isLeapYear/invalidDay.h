/*
   Author: William Wadsworth
   Date: 3.25.21
   Class: CSC1720
   Code location: ~/csc1720/lab10/invalidDay.h

   About:
      This is the header and implementation file for the invalidDay class
*/

#include <iostream>
using namespace std;

class invalidDay
{
   private:
      string message;
   public:
      // constructor
      invalidDay()
      {
         message = "invalid day";
      }
      // parameterized constructor
      invalidDay(string str)
      {
         message = str;
      }
      /* function huh returns the message from the invalidDay class
       * precondition: message is declared and initialized
       * postcondition: message is returned to the main program
       */
      string huh()
      {
         return message;
      }
};
