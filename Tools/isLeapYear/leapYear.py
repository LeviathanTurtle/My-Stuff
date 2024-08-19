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
import invalidDay
import invalidMonth
import invalidYear

# --- FUNCTIONS -------------------------------------------------------------------------
# --- IS LEAP -----------------------------------
# takes current year and checks if it is a leap year
def isLeap(year) -> bool:
    if year%4 == 0:
        return True
    else:
        return False

# --- CONV --------------------------------------
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


def main():
    # --- INPUT ---------------------------------
    try:
        day = int(input("Enter your birth day: "))
        month = int(input("Enter your birth month: "))
        year = int(input("Enter your birth year: "))

        # --- TEST INPUT ------------------------
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

        # --- CONVERSION ------------------------
        # convert numerical month to string
        print(f"{conv(month)}day, {year}")

        # output if the year is a leap year or not
        if isLeap(year):
            print("Your birth year is a leap year")
        else:
            print("Your birth year is not a leap year")


    # --- EXCEPTION HANDLING --------------------
    # date is not valid
    except invalidDay:
        print("error: ",invalidDay.huh())

    # month is not valid
    except invalidMonth:
        print("error: ",invalidMonth.huh())

    # year is not valid
    except invalidYear:
        print("error: ",invalidYear.huh())


if __name__ == "__main__":
    main()