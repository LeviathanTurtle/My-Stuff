/*
   Author: William Wadsworth
   Date: 4.5.21
   Class: CSC1720
   Code location: ~/csc1720/prog4/invalidBase.h

   About:
      This is the header file for the invalidBase class
*/

#ifndef invalidBase_h
#define invalidBase_h

#include <iostream>
using namespace std;

class invalidBase
{
    private:
        string message;
    public:
        invalidBase()
        {
            message = "the base must be positive";
        }
        invalidBase(string str)
        {
            message = str;
        }
        string huh()
        {
            return message;
        }
};

#endif
