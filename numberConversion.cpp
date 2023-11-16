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
bool searchCharacter(const char*);

// function to convert integers to scientific notation
string intToString(int);
// function to convert integers to scientific notation - from file
string* intToString(ifstream&);

// function to convert numbers in scientific notation to ingegers
int stringToInt(string);
// function to convert numbers in scientific notation to ingegers - from file
int* stringToInt(istream&);


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


    // if only the exe arg was provided, prompt user for number
    string number;
    // flags for types of numbesr
    bool isDecimal = false, isInt = false;
    if(argc == 1) {
        cout << "No number provided, you must enter one.\nNote: entering a "
             << "number in scientific notation will be converted to integer "
             << "form. If entering an exponent, it must be a whole number.\n"
             << "Enter a number: ";
        cin >> number;

        // if number is a decimal, mark respective flag
        if(searchCharacter(number,'.'))
            isDecimal = true;
        else
            isInt = true;
    }


    // process command line
    if(argv[1] == "-f") {
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
    else {
        // no file is specified
        
        // is argv[1] a string or int?
        // if string
        //     stringToInt
        // else
        //     intToString
        
        // we know it's a string if e or E is in argv[1]
        if(searchCharacter(argv[1],'e') || searchCharacter(argv[1],'E'))
            int answer = stringToInt(number); // contains e or E
        else
            string answer = intToString(stoi(number));
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
    if(searchCharacter(file.peek()))
        stringToInt(file,promptForItem);
    else
        intToString(file,promptForItem);

    // close file
    file.close();
}


// function to search for a specific character if a string (number in scientific notation)
// is provided -- by ChatGPT
bool searchCharacter(const char* string, char target)
{
    // iterate through string
    for(char ch : string)
        if(ch == target)
            return true; // found target
    
    // target not found
    return false;
}


// function to convert integers to scientific notation
string intToString(int number)
{

}


// function to convert integers to scientific notation
// CONVERT FROM FILE
string* intToString(ifstream& file)
{
    // while there is input, get and process item

}


// function to convert numbers in scientific notation to ingegers
int stringToInt(string number)
{
    int num;

    // if number was not provided, get from user
    if(!promptForItem) {
        cout << "enter a number (in scientific notation): ";
        cin >> num;
    }
    else 
        num = number;
    
    // numbers left of E = L
    // numbers right of E = R

    // L * 10^R


}


// function to convert numbers in scientific notation to ingegers
// CONVERT FROM FILE
int* stringToInt(ifstream& file)
{
    // while there is input, get and process item

}