/* PERSONAL COFFEE SALES STATISTICS
 * William Wadsworth
 * Created: 9.16.2020
 * Doctored: 11.2.2023
 * CSC 1710
 * ~/csc1710/lab5/lab5.cpp
 *
 *
 * [DESCRIPTION]:
 * This program calculates and outputs information pertaining to coffee sales based on an input 
 * file. The binary was last compiled on 5.24.2024.
 *
 *
 * [COMPILE/RUN]:
 * To compile:
 *     g++ personalSales.cpp -Wall -o personalSales
 *
 * To run (3 args):
 *     ./personalSales <input file> <output file>
 *
 *
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed full execution
 *
 * 1 - CLI args used incorrectly
 *
 * 2 - file unable to be opened or created
*/

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
using namespace std;


struct personalSales {
    string firstName;
    string lastName;
    string department;
    double salary;
    double bonus;
    double taxes;
    double distance;
    double time;
    int cups;
    double cupCost;
};


personalSales loadStruct(char*);
void dump(char*, char*);


int main (int argc, char* argv[])
{
    // check CLI args are used correctly
    if(argc != 3) {
        cerr << "error: CLI args used incorrectly. Proper order: ./exe <input "
             << "file> <output file>\n";
        exit(1);
    }

    // ------------------------------------------------------------------------
    // INPUT

    // initialize variables
    personalSales person = loadStruct(argv[1]);

    // ------------------------------------------------------------------------
    // OUTPUT

    dump(argv[1],argv[2]);

    return 0;
}


personalSales loadStruct(char* inputFile)
{
    personalSales person;

    // declare data file variables
    ifstream file (inputFile);

    // check file was opened
    if(!file) {
        cerr << "error: file unable to be opened or created.\n";
        exit(2);
    }

    // take first and last values from data file, display in output
    file >> person.firstName;
    file >> person.lastName;
    file >> person.department;
    cout << "Name: " << person.firstName << " " << person.lastName << ", Department: "
         << person.department << endl;

    // take salary, bonus, and tax values from data file, display in output, 
    // set to display 2 decimal values
    cout << fixed << showpoint << setprecision(2);
    file >> person.salary;
    file >> person.bonus;
    file >> person.taxes;
    cout << "Monthly Gross Income: $" << person.salary << ", Bonus: " 
         << person.bonus << "%, Taxes: " << person.taxes << "%" << endl;

    // take distance and time values from data file, display in output, 
    // calculate mph
    file >> person.distance;
    file >> person.time;
    cout << "Distance traveled: " << person.distance << " miles, "
         << "Traveling Time: " << person.time << " hours" << endl;
    cout << "Average Speed: " << person.distance/person.time << " miles per "
         << "hour" << endl;

    // take cups and cost values from data file, display in output
    file >> person.cups;
    file >> person.cupCost;
    cout << "Number of coffee cups sold: " << person.cups << ", Cost: $" 
         << person.cupCost << " per cup" << endl;
    cout << "Sales amount = $" << person.cups*person.cupCost << endl;

    // close input file
    file.close();

    return person;
}


void dump(char* inputFileName, char* outputFileName)
{
    ifstream inputFile (inputFileName);
    ofstream outputFile (outputFileName);

    // check files were opened
    if(!inputFile || !outputFile) {
        cerr << "error: file(s) unable to be opened or created. provided names"
             << ": input: " << inputFileName << ", output: " << outputFileName 
             << ".\n";
        exit(2);
    }

    // read output into the output file
    char ch;
    while (inputFile.get(ch))
        outputFile.put(ch);

    // close files
    inputFile.close();
    outputFile.close();
}

