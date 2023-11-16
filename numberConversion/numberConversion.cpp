/* CONVERTING INTEGERS TO STRINGS (WITH SCIENTIFIC NOTATION)
 * William Wadsworth
 * Created: 11.16.2023
 * 
 * 
 * [DESCRIPTION]:
 * This program converts integers to strings (and vice versa), accounting for
 * scientific notation for numbers greater than 1,000. The number can be proved
 * as input, passed as an argument in program execution, or a file of numbers
 * can be provided.
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ numberConversion.cpp -Wall -o numberConversion
 * 
 * To run (5 args, 4 optional):
 *     ./numberConversion [-f] [file] [number] [string]
 * 
 * where:
 *     [-f]     - optional flag, specify if you are providing a file
 *     [file]   - optional, but required if -f is specified, file containing
 *                numbers in or not in scientific notation to convert
 *     [number] - optional, provide a number to convert
 *     [string] - optional, provide a number in scientific notation
 * 
 * Note: if providing a second argument, it must either be a number or a file.
 * 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - CLI args used incorrectly
 * 
 * 2 - file not provided (-f is specified)
 * 
 * 3 - file unable to be opened
*/

#include <iostream>
#include <fstream>
using namespace std;

// FUNCTION PROTOTYPES
// ----------------------------------------------------------------------------

// function to process file, calls string or int conversion
void processFile(const char*);

// function to search for specific character if a string (number in scientific
// notation) is provided -- by ChatGPT
bool searchCharacter(const char*, const char&);

// function to update the flags for the type of the output number
void determineType(const string&, bool&, bool&, bool&);

// convert SCIENTIFIC NOTATION to INTEGER
int stringToInt(const string&);    // 3e3
// FILE overload
int* stringToInt(const string&);

// convert SCIENTIFIC NOTATION to DECIMAL
double stringToDouble(const string&); // 3e-3
// FILE overload
double* stringToDouble(const string&);

// convert INTEGERS to SCIENTIFIC NOTATION
string intToString(const int&);       // 128000
// FILE overload
string* intToString(const int&);

// convert DECIMAL to SCIENTIFIC NOTATION
string doubleToString(const double&); // .003
// FILE overload
string* doubleToString(const double&);


// MAIN CODE
// ----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // check optional CLI args
    // only checking upper bound - 1 is required, less than 1 arg is 0 args 
    // provided which means nothing. Max of 3 args
    if(argc > 3) {
        cerr << "error: CLI args used incorrectly: ./numberConversion [-f] "
             << "[file] [number] [string].\n\nwhere:\n"
             << "    [-f]     - optional flag, specify if you are providing a file\n"
             << "    [file]   - optional, but required if -f is specified, file\n"
             << "               containing numbers in or not in scientific notation to convert\n"
             << "    [number] - optional, provide a number to convert\n"
             << "    [string] - optional, provide a number in scientific notation\n";
        exit(1);
    }


    // flags for types of number (if needed to be input by user)
    // separate flags for integer and decimal because taking a float/double as
    // an integer is truncated
    bool isDecimal = false, isInt = false, isExponent = false;


    // number was not provided at execution
    if(argc == 1) {
        // if only the exe arg was provided, prompt user for number
        string number;
        
        cout << "No number provided, you must enter one.\nNote: entering a "
             << "number in scientific notation will be converted to integer "
             << "form. If entering an exponent, it must be a whole number.\n"
             << "Enter a number: ";
        cin >> number;

        // set respective flag
        determineType(number,isDecimal,isInt,isExponent);

        // if integer
        // --> string to int
        // else if decimal
        // --> string to double
        // else 
        // --> 

        // call respective functions
        
    }
    // file was provided at execution
    else if(argv[1] == "-f") {
        // -f specified, file MUST follow
        
        // call function for file, which calls its respective function
        if(argc == 3)
            processFile(argv[2]);
        // error for if filename is not provided
        else {
            cerr << "error: file not provided.\n";
            exit(2);
        }
    }
    // number was provided at execution
    else {
        // no file is specified
        
        // is argv[1] a string or int?
        // if string
        //     stringToInt
        // else
        //     intToString
        
        // set respective flag
        determineType(argv[1],isDecimal,isInt,isExponent);

        // same approach as above

    }
    


    return 0;
}


// function to process file, calls string or int conversion
void processFile(const char* filename)
{
    ifstream file (filename);

    if(!file) {
        cerr << "error: file (" << filename << ") unable to be opened.\n";
        exit(3);
    }

    // check first value in file
    // if it contains e or E --> stringToInt
    // else --> intToString
    determineType(file.peek(),isDecimal,isInt,isExponent);
    



    // close file
    file.close();
}


// function to search for a specific character if a string (number in scientific notation)
// is provided -- by ChatGPT
bool searchCharacter(const char* string, const char& target)
{
    // iterate through string
    for(char ch : string)
        if(ch == target)
            return true; // found target
    
    // target not found
    return false;
}


// function to determine if an exponent is negative
bool searchCharacter(const char* string)
{
    bool isRight = false;

    // iterate through string
    for(char ch : string) {
        if(ch == 'e' || ch == 'E')
            isRight = true;        

        if(isRight && ch == '-')
            return true; // found target
    }
    
    // target not found
    return false;
}


// function to update the flags for the type of the output number
void determineType(const string& number, bool& isDecimal, bool& isInt, bool& isExponent)
{
    // if the number is in scientific notation, mark respective flag
    if(searchCharacter(number,'e') || searchCharacter(number,'E'))
        isExponent = true;
    // if number is a decimal, mark respective flag
    else if(searchCharacter(number))
        isDecimal = true;
    // assume number is an integer
    else
        isInt = true;
}


// INTEGERS to SCIENTIFIC NOTATION
// ----------------------------------------------------------------------------

string intConv(const int& number)
{

}

// FILE overload
string* intConv(const int& number)
{

}

// DECIMAL to SCIENTIFIC NOTATION
// ----------------------------------------------------------------------------

string doubleConv(const double&)
{

}

// FILE overload
string* doubleConv(const double&)
{

}

// SCIENTIFIC NOTATION to INTEGER
// ----------------------------------------------------------------------------

int stringConv(const string&)
{

}


// FILE overload
int* stringConv(const string&)
{

}

// SCIENTIFIC NOTATION to DECIMAL
// ----------------------------------------------------------------------------

double stringConv(const string&)
{

}

// FILE overload
double* stringConv(const string&)
{

}

