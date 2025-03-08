/* IS YOUR BIRTHDAY A LEAP YEAR
 * Author: William Wadsworth
 * Created: 3.25.2021
 * Doctored: 10.25.2023
 * 
 * Class: CSC1720
 * ~/csc1720/lab10/wadsworthLab10.cpp
 * 
 * 
 * [DESCRIPTION]:
 * This program prompts the user for their birthday, checks if the date is
 * valid, and displays if the birth year is a leap year.
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ leapYear.cpp -Wall -o leapYear
 * 
 * To execute:
 *     ./leapYear
*/

#include <iostream>
#include "invalidCalendar.h"
using namespace std;

// function prototypes
bool isLeap(int);
void conv(int);

int main()
{
    // 3 variables
    int day, month, year;
    try {
        // birth day
        cout << "Enter your birth day: ";
        cin >> day;

        // birth month
        cout << "Enter your birth month (numerical): ";
        cin >> month;
        
        // birth year
        cout << "Enter your birth year: ";
        cin >> year;
        cout << endl;
        
        // INPUT TESTING
        // year test
        //if (year <= 1582)
        //    throw invalidYear();
        
        // month test
        if (month <= 0 || month > 12)
            throw invalidMonth();
        
        // day test
        switch (month) {
            // months with 31 days
            case 1:
            case 3:
            case 5:
            case 7:
            case 8:
            case 10:
            case 12:
                if (day <= 0 || day > 31)
                    throw invalidDay();

            // february
            case 2:
                if (day <= 0)
                    throw invalidDay();
                if (day > 28 && !isLeap(year))
                    throw invalidDay();
                if (day > 29 && isLeap(year))
                    throw invalidDay();

            // months with 30 days
            case 4:
            case 6:
            case 9:
            case 11:
                if (day <= 0 || day > 30)
                    throw invalidDay();
        }

        // convert numerical month to string
        conv(month);
        cout << " " << day << ", " << year << endl;
        // output if the year is a leap year or not
        if (isLeap(year))
            cout << "Your birth year is a leap year" << endl;
        else
            cout << "Your birth year is not a leap year" << endl;
    }
    // date is not valid
    catch (invalidDay invDay) {
        cout << "error: " << invDay.huh() << endl;
    }
    // month is not valid
    catch (invalidMonth invMonth) {
        cout << "error: " << invMonth.huh() << endl;
    }
    // year is not valid
    catch (invalidYear invYear) {
        cout << "error: " << invYear.huh() << endl;
    }

    return 0;
}

// takes current year and checks if it is a leap year
bool isLeap(int year)
{
    if (year % 4 == 0)
        return true;
    else
        return false;
}

// takes numeric month input and outputs corresponding month
void conv(int inp)
{
    switch (inp)
    {
        case 1:
            cout << "January";
            break;
        case 2:
            cout << "February";
            break;
        case 3:
            cout << "March";
            break;
        case 4:
            cout << "April";
            break;
        case 5:
            cout << "May";
            break;
        case 6:
            cout << "June";
            break;
        case 7:
            cout << "July";
            break;
        case 8:
            cout << "August";
            break;
        case 9:
            cout << "September";
            break;
        case 10:
            cout << "October";
            break;
        case 11:
            cout << "November";
            break;
        case 12:
            cout << "December";
            break;
    }
}
