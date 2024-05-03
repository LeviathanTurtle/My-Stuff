# this is out of date

The programs in this directory were originally written in C++. They were later 
translated to Rust and Python.

[PROGRAMS]:
compAlgSort: loads array of values from datafile, runs and runs one of: [bubble
             (v1), bubble (v2), merge] sorting algorithms. Records time elapsed.



[TOOLS]:
create_test: creates a test file consisting of random datatypes (integer,
             double, float, character, string), named accordingly based on the
             number of test files in the TestFiles directory. The user can
             specify if they want the output to be in matrix form. Matrix 
             output is limited to numerical datatypes and characters.

sizeOfFile: determines the amount of entries in a file passed by argv. The
            program assumes each data value is separated by a new line. It also
            outputs the time elapsed to count the entries using sys/time.h.

hash.cpp: example of hashing a value.

dataDetails: loads data from a file. Data termination is marked by a value of 
             -1 in the files. The program outputs the raw array with each 
             element and its corresponding value, sorts it, then outputs the
             sorted array with the same structure. It then outputs the median,
             minumum, maximum, and average values with their locations.
             Afterwards, the user is prompted to search the array for a number
             and count the occurences.



[CALCULATORS]:
fourFunction: simple, 4 function calculator.

isMultiple: check if two numbers are multiples, and how many multiples if
            applicable.

moneyCalculator: calculate sum of money based on amount of dollar bills.

factorialGeoseriesCalc: calculate factorial, double factorial, or geometric 
                        series of an input number.

finalGrade: calculate the final grade of a course. Prompts user for grade
            weights and calculates total.
        
investTable: creates an investment table based on user input. The user inputs
             their principle amount, the interest rate (and the rate of change
             per month if it changes), the amount of time elapsed, and their
             monthly deposit amount.

leapYear: prompts the user for their birthday and outputs if it is a leap year.

coinTotal: prompts the user to input a price, and the program will calculate
           and output the minimum amount of coins for each type (quarter, dime,
           nickel, penny) required to meet the price.

isRightTriangle: prompts the user for 3 sides, outputs if it makes a right 
                 triangle or not.

tempConv: the user is prompted for the upper and lower bound of a temperature
          range in Fahrenheit, as well as how much they want to increment each
          step by. The program starts the output with their lower bound, and 
          its value in Celsius and Kelvin. This is repeated for each increment
          value until it reaches the upper bound.



[GAMES]:
hangman

