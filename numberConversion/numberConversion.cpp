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
 *     ./numberConversion [-fD] [file] [number] [string]
 * 
 * where:
 *     [-f]     - optional flag, specify if you are providing a file
 *     [-D]     - optional flag, specify debug (verbose) output
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
 * 
 * 4 - not valid flag
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
void stringToInt(const string&);    // 3e3
// FILE overload
void stringToInt(ifstream&);

// convert SCIENTIFIC NOTATION to DECIMAL
void stringToDouble(const string&); // 3e-3
// FILE overload
void stringToDouble(iftream&);

// convert INTEGERS to SCIENTIFIC NOTATION
void intToString(const int&);       // 128000
// FILE overload
void intToString(ifstream&);

// convert DECIMAL to SCIENTIFIC NOTATION
void doubleToString(const double&); // .003
// FILE overload
void doubleToString(ifstream&);


// variable for verbose output
bool DEBUG = false;


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


    // check flags
    if(argc > 1) {
        // verbose output was provided at execution
        if(searchCharacter(argv[1], '-'))
            if(searchCharacter(argv[1], 'D'))
                DEBUG = true;
        // file was provided at execution
        else if(searchCharacter(argv[1], 'f')) {
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
        // error - not valid flag
        else {
            cerr << "error: not a valid flag.\n";
            exit(4);
        }
    }


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
        // --> int to string
        // else if decimal
        // --> double to string

        // else (string) 
        // --> convert to int?
        // --> or convert to double

        // call respective functions
        if(isInt)
            intToString(atoi(number)); // the number is an integer (128000)
        else if(isDecimal)
            // atod? stod, stof
            doubleToString(atof(number)); // the number is a decimal (.128)
        // SCIENTIFIC NOTATION WAS GIVEN
        else {
            //if(isExponent)
            //    if(searchCharacter(number,'-'))
            //        stringToDouble(number); // the string is a decimal (128e-3)
            if(isExponent && searchCharacter(number,'-'))
                stringToDouble(number); // the string is a decimal (128e-3)
            else
                stringToInt(number); // the string is an integer (128e3)
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
        // call respective functions
        if(isInt)
            intToString(atoi(argv[1])); // the number is an integer (128000)
        else if(isDecimal)
            // atod? stod, stof
            doubleToString(atof(argv[1])); // the number is a decimal (.128)
        // SCIENTIFIC NOTATION WAS GIVEN
        else {
            if(isExponent && searchCharacter(argv[1],'-'))
                stringToDouble(argv[1]); // the string is a decimal (128e-3)
            else
                stringToInt(argv[1]); // the string is an integer (128e3)
        }
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

    // same approach as main -- call respective functions
    // call respective functions
    if(isInt)
        intToString(file); // the number is an integer (128000)
    else if(isDecimal)
        // atod? stod, stof
        doubleToString(file); // the number is a decimal (.128)
    // SCIENTIFIC NOTATION WAS GIVEN
    else
        if(isExponent && searchCharacter(file.peek(),'-'))
            stringToDouble(file); // the string is a decimal (128e-3)
        else
            stringToInt(file); // the string is an integer (128e3)

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

void intToString(const int& number)
{

}

// FILE overload
void intToString(ifstream& file)
{
    // repeat for every entry in the file
    while(!file.eof()) {

    }
}

// DECIMAL to SCIENTIFIC NOTATION
// ----------------------------------------------------------------------------

void doubleToString(const double&)
{

}

// FILE overload
void doubleToString(ifstream& file)
{
    // repeat for every entry in the file
    while(!file.eof()) {
        
    }
}

// SCIENTIFIC NOTATION to INTEGER
// ----------------------------------------------------------------------------

void stringToInt(const string&)
{
    // variable for help getting value after e
    bool isRight = false;


}


// FILE overload
void stringToInt(ifstream& file)
{
    // repeat for every entry in the file
    while(!file.eof()) {
        
    }
}

// SCIENTIFIC NOTATION to DECIMAL
// ----------------------------------------------------------------------------

void stringToDouble(const string&)
{

}

// FILE overload
void stringToDouble(ifstream& file)
{
    // repeat for every entry in the file
    while(!file.eof()) {
        
    }
}

