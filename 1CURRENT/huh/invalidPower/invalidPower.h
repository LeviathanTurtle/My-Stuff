/*
   Author: William Wadsworth
   Date: 4.5.21
   Class: CSC1720
   Code location: ~/csc1720/prog4/invalidPower.h

   About:
      This is the header file for the invalidPower class
*/

#ifndef invalidPower_h
#define invalidPower_h

#include <iostream>
using namespace std;

class invalidPower
{
    private:
        string message;
    public:
        invalidPower()
        {
            message = "the power must be positive";
        }
        invalidPower(string str)
        {
            message = str;
        }
        string huh()
        {
            return message;
        }
};

#endif
