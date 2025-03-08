/*
 * William Wadsworth
 * 3.25.21
 * CSC1720
 *
 * This is the header and implementation file for the invalidDay, invalidMonth, and invalidYear
 * classes.
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

        /* returns the message from the invalidDay class
        * pre-condition: message is declared and initialized
        * post-condition: message is returned to the main program
        */
        string huh()
        {
            return message;
        }
};

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

        /* returns the message from the invalidMonth class
        * pre-condition: message is declared and initialized
        * post-condition: message is returned to the main program
        */
        string huh()
        {
            return message;
        }
};

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

        /* returns the message from the invalidYear class
         * pre-condition: message is declared and initialized
         * post-condition: message is returned to the main program
         */
        string huh()
        {
            return message;
        }
};

