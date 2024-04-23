/*
   Author: William Wadsworth
   Date: 3.25.21
   Class: CSC1720
   Code location: ~/csc1720/lab10/invalidYear.h

   About:
      This is the header and implementation file for the invalidYear class
*/

#include <iostream>
using namespace std;

class invalidYear
{
    private:
        string message;
    public:
        // constructor
        invalidYear()
        {
            message = "invalid year";
        }
        // parameterized constructor
        invalidYear(string str)
        {
            message = str;
        }
        /* function huh returns the message from the invalidYear class
         * precondition: message is declared and initialized
         * postcondition: message is returned to the main program
         */
        string huh()
        {
            return message;
        }
};
