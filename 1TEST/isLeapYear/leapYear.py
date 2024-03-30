# IS YOUR BIRTHDAY A LEAP YEAR -- V.PY
# William Wadsworth
# CSC1710
# Created: 3.25.2021
# Doctored: 10.25.2023
# Python-ized: 3.30.2024
# 
# [DESCRIPTION]: 
# This program prompts the user for their birthday, checks if the date is valid, and displays if
# the birth year is a leap year.
# 
# [USAGE]:
# python3 leapYear.py 


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include "invalidDay.h"
#include "invalidMonth.h"
#include "invalidYear.h"
using namespace std;
"""
import invalidDay
import invalidMonth
import invalidYear

# --- FUNCTIONS -------------------------------------------------------------------------
# --- IS LEAP -----------------------------------
"""
bool isLeap(int);
bool isLeap(int year)
{
    if (year % 4 == 0)
        return true;
    else
        return false;
}
"""
# takes current year and checks if it is a leap year
def isLeap(year) -> bool:
    if year%4 == 0:
        return True
    else:
        return False

# --- CONV --------------------------------------
"""
void conv(int);
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
"""
# takes numeric month input and outputs corresponding month
# end="" --> not add newline to the end of the string
# sep="" --> not add space between function args
def conv(inp):
    match(inp):
        case 1:
            print("January",end="")
        case 2:
            print("February",end="")
        case 3:
            print("March",end="")
        case 4:
            print("April",end="")
        case 5:
            print("May",end="")
        case 6:
            print("June",end="")
        case 7:
            print("July",end="")
        case 8:
            print("August",end="")
        case 9:
            print("September",end="")
        case 10:
            print("October",end="")
        case 11:
            print("November",end="")
        case 12:
            print("December",end="")

# --- MAIN ------------------------------------------------------------------------------
# --- INPUT -------------------------------------
"""
int main()
{
    int day, month, year;
    try {
        cout << "Enter your birth day: ";
        cin >> day;

        cout << "Enter your birth month (numerical): ";
        cin >> month;
        
        cout << "Enter your birth year: ";
        cin >> year;
        cout << endl;
"""
try:
    day = int(input("Enter your birth day: "))
    month = int(input("Enter your birth month: "))
    year = int(input("Enter your birth year: "))

# --- TEST INPUT --------------------------------
    """
        if (year <= 1582)
            throw invalidYear();
        
        if (month <= 0 || month > 12)
            throw invalidMonth();
        
        switch (month) {
            case 1:
            case 3:
            case 5:
            case 7:
            case 8:
            case 10:
            case 12:
                if (day <= 0 || day > 31)
                    throw invalidDay();

            case 2:
                if (day <= 0)
                    throw invalidDay();
                if (day > 28 && !isLeap(year))
                    throw invalidDay();
                if (day > 29 && isLeap(year))
                    throw invalidDay();

            case 4:
            case 6:
            case 9:
            case 11:
                if (day <= 0 || day > 30)
                    throw invalidDay();
        }
    """
    # year test
    if year <= 0:
        raise invalidYear()
    # month test
    if month <= 0 or month > 12:
        raise invalidMonth()
    # day test -- months with 31 days
    if month in {1,3,5,7,8,10,12}:
        if day <= 0 or day > 31:
            raise invalidDay()
    # day test -- february
    elif month == 2:
        if day <= 0:
            raise invalidDay()
        if day > 28 and not isLeap(year):
            raise invalidDay()
        if day > 29 and isLeap(year):
            raise invalidDay()
    # day test -- months with 30 days
    elif month in {4, 6, 9, 11}:
        if day <= 0 or day > 30:
            raise invalidDay()

# --- CONVERSION --------------------------------
    """
        conv(month);
        cout << " " << day << ", " << year << endl;
        
        if (isLeap(year))
            cout << "Your birth year is a leap year" << endl;
        else
            cout << "Your birth year is not a leap year" << endl;
    }
    """
    # convert numerical month to string
    print(f"{conv(month)}day, {year}")

    # output if the year is a leap year or not
    if isLeap(year):
        print("Your birth year is a leap year")
    else:
        print("Your birth year is not a leap year")


# --- EXCEPTION HANDLING ------------------------
    """
    catch (invalidDay invDay) {
        cout << "error: " << invDay.huh() << endl;
    }
    catch (invalidMonth invMonth) {
        cout << "error: " << invMonth.huh() << endl;
    }

    catch (invalidYear invYear) {
        cout << "error: " << invYear.huh() << endl;
    }

    return 0;
}
    """
# date is not valid
except invalidDay:
    print("error: ",invalidDay.huh())

# month is not valid
except invalidMonth:
    print("error: ",invalidMonth.huh())

# year is not valid
except invalidYear:
    print("error: ",invalidYear.huh())

