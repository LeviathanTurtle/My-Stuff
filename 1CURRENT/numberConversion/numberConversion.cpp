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
// math
#include <cmath>
#include <iomanip>
using namespace std;

// FUNCTION PROTOTYPES
// ----------------------------------------------------------------------------

// function to process file, calls string or int conversion
void processFile(const char*);

// function to search for specific character if a string (number in scientific
// notation) is provided -- by ChatGPT
bool searchCharacter(const string&, const char&);
// FILE overload
bool searchCharacter(const string&);

// function to update the flags for the type of the output number
int determineType(const string&);

// convert SCIENTIFIC NOTATION to INTEGER
int stringToInt(const string&);    // 3e3
// FILE overload
void stringToInt(ifstream&, const string&);

// convert SCIENTIFIC NOTATION to DECIMAL
double stringToDouble(const string&); // 3e-3
// FILE overload
void stringToDouble(ifstream&, const string&);

// convert INTEGERS to SCIENTIFIC NOTATION
string intToString(const int&);       // 128000
// FILE overload
void intToString(ifstream&, const string&);

// convert DECIMAL to SCIENTIFIC NOTATION
string doubleToString(const double&); // .003
// FILE overload
void doubleToString(ifstream&, const string&);


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
        int result = determineType(number);

        // if integer
        // --> int to string
        // else if decimal
        // --> double to string

        // else (string) 
        // --> convert to int?
        // --> or convert to double

        // call respective functions
        switch(result) {

        }
        if(isInt) {
            // move elsewhere?
            cout << fixed << setprecision(1);
            cout << intToString(stoi(number)) << endl; // the number is an integer (128000)
        }
        else if(isDecimal) {
            // atod? stod, stof
            cout << fixed << setprecision(1);
            cout << doubleToString(stof(number)) << endl; // the number is a decimal (.128)
        }
        // SCIENTIFIC NOTATION WAS GIVEN
        else {
            //if(isExponent)
            //    if(searchCharacter(number,'-'))
            //        stringToDouble(number); // the string is a decimal (128e-3)
            if(isExponent && searchCharacter(number,'-'))
                cout << stringToDouble(number) << endl; // the string is a decimal (128e-3)
            else
                cout << stringToInt(number) << endl; // the string is an integer (128e3)
        }
    }
    // number was provided at execution
    else {
        // no file is specified
        // CONVERTING ONE NUMBER
        
        // is argv[1] a string or int?
        // if string
        //     stringToInt
        // else
        //     intToString
        
        // set respective flag
        int result = determineType(argv[1]);

        // same approach as above
        // call respective functions
        switch(result) {

        }
        if(isInt) {
            cout << fixed << setprecision(1);
            cout << intToString(atoi(argv[1])) << endl; // the number is an integer (128000)
        }
        else if(isDecimal) {
            // atod? stod, stof
            cout << fixed << setprecision(1);
            cout << doubleToString(atof(argv[1])) << endl; // the number is a decimal (.128)
        }
        // SCIENTIFIC NOTATION WAS GIVEN
        else {
            if(isExponent && searchCharacter(argv[1],'-'))
                cout << stringToDouble(argv[1]) << endl; // the string is a decimal (128e-3)
            else
                cout << stringToInt(argv[1]) << endl; // the string is an integer (128e3)
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
    int result = determineType(to_string(file.peek()));

    // same approach as main -- call respective functions
    // call respective functions
    switch(result) {
        case 1:
            intToString(file, filename); // the number is an integer (128000)
        case 2:
            // atod? stod, stof
            doubleToString(file, filename); // the number is a decimal (.128)
        
        // SCIENTIFIC NOTATION WAS GIVEN
        case 3:
            if(searchCharacter(to_string(file.peek()),'-'))
                stringToDouble(file, filename); // the string is a decimal (128e-3)
            else
                stringToInt(file, filename); // the string is an integer (128e3)
    }   

    // close file
    file.close();
}


// function to search for a specific character if a string (number in scientific notation)
// is provided -- by ChatGPT
bool searchCharacter(const string& item, const char& target)
{
    // iterate through string
    for(char ch : item)
        if(ch == target)
            return true; // found target
    
    // target not found
    return false;
}


// function to determine if an exponent is negative
bool searchCharacter(const string& item)
{
    bool isRight = false;

    // iterate through string
    for(char ch : item) {
        if(ch == 'e' || ch == 'E')
            isRight = true;        

        if(isRight && ch == '-')
            return true; // found target
    }
    
    // target not found
    return false;
}


// function to update the flags for the type of the output number
int determineType(const string& number)
{
    // if the number is in scientific notation, mark respective flag
    if(searchCharacter(number,'e') || searchCharacter(number,'E'))
        return 3;
    // if number is a decimal, mark respective flag
    else if(searchCharacter(number))
        return 2;
    // assume number is an integer
    else
        return 1;
}


// INTEGERS to SCIENTIFIC NOTATION
// ----------------------------------------------------------------------------

string intToString(const int& number)
{
    // check if number is 0
    if(number == 0) {
        cout << "0" << endl;
        return;
    }
    
    // find exponent
    int exponent = static_cast<int>(log10(abs(number)));
    // any problems with static_cast?

    // find base ("mantissa")
    double base = number / pow(10, exponent);

    // return answer 
    return to_string(base) + 'e' + to_string(exponent);
}

// FILE overload
void intToString(ifstream& file, const string& filename)
{
    // create output file object
    ofstream outputFile (filename+"-output");

    // holder variable for current value in filee
    int temp;
    
    // repeat for every entry in the file
    //while(!file.eof()) {
    while(file >> temp)
        outputFile << intToString(temp);

    outputFile.close();
}

// DECIMAL to SCIENTIFIC NOTATION
// ----------------------------------------------------------------------------

string doubleToString(const double&)
{

}

// FILE overload
void doubleToString(ifstream& file, const string& filename)
{
    // repeat for every entry in the file
    while(!file.eof()) {
        
    }
}

// SCIENTIFIC NOTATION to INTEGER
// ----------------------------------------------------------------------------

int stringToInt(const string&)
{
    // variable for help getting value after e
    bool isRight = false;


}


// FILE overload
void stringToInt(ifstream& file, const string& filename)
{
    // repeat for every entry in the file
    while(!file.eof()) {
        
    }
}

// SCIENTIFIC NOTATION to DECIMAL
// ----------------------------------------------------------------------------

double stringToDouble(const string&)
{
    // variable for help getting value after e
    bool isRight = false;


}

// FILE overload
void stringToDouble(ifstream& file, const string& filename)
{
    // repeat for every entry in the file
    while(!file.eof()) {
        
    }
}

