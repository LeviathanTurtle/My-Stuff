/*
   Author: William Wadsworth
   Date: 4.5.21
   Class: CSC1720
   Code location: ~/csc1720/prog4/overflow.h

   About:
      This is the header file for the overflow class
*/

#ifndef overflow_h
#define overflow_h

#include <iostream>
using namespace std;

class overflow
{
    private:
        string message;
    public:
        overflow()
        {
            message = "value exceeds integer limit";
        }
        overflow(string str)
        {
            message = str;
        }
        string huh()
        {
            return message;
        }
};

#endif
