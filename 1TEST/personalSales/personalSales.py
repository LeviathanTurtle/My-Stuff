# PERSONAL COFFEE SALES STATISTICS -- V.PY
# William Wadsworth
# CSC1710
# Created: 9.16.2020
# Doctored: 11.2.2023
# Python-ized: 3.30.2024
# 
# [DESCRIPTION]:
# This program calculates and outputs information pertaining to coffee sales based on an input file
# 
# [USAGE]:
# python3 personalSales.py <input file> <output file>
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed full execution
# 
# 1 - CLI args used incorrectly
# 
# 2 - file unable to be opened or created


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
using namespace std;
"""
import sys
from dataclasses import dataclass

# --- OBJECTS ---------------------------------------------------------------------------
"""
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
"""
@dataclass
class personalSales:
    first_name: str
    last_name: str
    department: str
    salary: float
    bonus: float
    taxes: float
    distance: float
    time: float
    cups: int
    cup_cost: float

# --- FUNCTIONS -------------------------------------------------------------------------
# --- LOAD STRUCT -------------------------------
"""
personalSales loadStruct(char*);
personalSales loadStruct(char* inputFile)
{
    personalSales person;

    ifstream inputFile (inputFile);

    if(!inputFile) {
        cerr << "error: file unable to be opened or created.\n";
        exit(2);
    }

    inputFile >> person.firstName;
    inputFile >> person.lastName;
    inputFile >> person.department;
    cout << "Name: " << person.first << " " << person.last << ", Department: "
         << person.department << endl;

    cout << fixed << showpoint << setprecision(2);
    inputFile >> person.salary;
    inputFile >> person.bonus;
    inputFile >> person.taxes;
    cout << "Monthly Gross Income: $" << person.salary << ", Bonus: " 
         << person.bonus << "%, Taxes: " << person.taxes << "%" << endl;

    inputFile >> person.distance;
    inputFile >> person.time;
    cout << "Distance traveled: " << person.distance << " miles, "
         << "Traveling Time: " << person.time << " hours" << endl;
    cout << "Average Speed: " << person.distance/person.time << " miles per "
         << "hour" << endl;

    inputFile >> person.cups;
    inputFile >> person.cupCost;
    cout << "Number of coffee cups sold: " << person.cups << ", Cost: $" 
         << person.cupCost << " per cup" << endl;
    cout << "Sales amount = $" << person.cups*person.cupCost << endl;

    inputFile.close();

    return person;
}
"""
def loadStruct(input_file) -> personalSales:
    person = personalSales()
    
    try:
        with open(input_file,'r') as file:            
            # take first and last values from data file, display in output
            person.firstName = input_file.readline().strip()
            person.lastName = input_file.readline().strip()
            person.department = input_file.readline().strip()
            print(f"Name: {person.firstName} {person.lastName}, Department: {person.department}")

            # take salary, bonus, and tax values from data file, display in output, set to display
            # two decimal places
            person.salary = float(input_file.readline())
            person.bonus = float(input_file.readline())
            person.taxes = float(input_file.readline())
            print(f"Monthly Gross Income: ${person.salary:.2f}, Bonus: {person.bonus:.2f}%, Taxes: {person.taxes:.2f}%")

            # take distance and time values from data file, display in output, calculate mph
            person.distance = float(input_file.readline())
            person.time = float(input_file.readline())
            print(f"Distance traveled: {person.distance:.2f} miles, Traveling Time: {person.time:.2f} hours")
            print(f"Average Speed: {person.distance/person.time:.2f} miles per hour")

            # take cups and cost values from data file, display in output
            person.cups = int(input_file.readline())
            person.cupCost = float(input_file.readline())
            print(f"Number of coffee cups sold: {person.cups}, Cost: ${person.cupCost:.2f} per cup")
            print(f"Sales amount = ${person.cups*person.cupCost:.2f}")
            
    except IOError:
        print(f"error: file (name: {input_file}) unable to be opened or created.")    
    
    return person    

# --- DUMP --------------------------------------
"""
void dump(char*, char*);
void dump(char* inputFileName, char* outputFileName)
{
    ifstream inputFile (inputFileName);
    ofstream outputFile (outputFileName);

    if(!inputFile || !outputFile) {
        cerr << "error: file(s) unable to be opened or created. provided names"
             << ": input: " << inputFileName << ", output: " << outputFileName 
             << ".\n";
        exit(2);
    }

    outputFile << inputFile;

    outputFile.close();
}
"""
def dump(input_file_name, output_file_name):
    # check files were opened
    try:
        with open(input_file_name, 'r') as input_file, open(output_file_name, 'w') as output_file:
            # read input file and write its contents to the output file
            output_file.write(input_file.read())
    except IOError:
        print(f"error: file(s) unable to be opened or created (input: {input_file_name}, output: {output_file_name}).")
        exit(2)

# --- MAIN ------------------------------------------------------------------------------
# --- CHECK CLI ARGS ----------------------------
"""
int main (int argc, char* argv[])
{
    if(argc != 3) {
        cerr << "error: CLI args used incorrectly. Proper order: ./exe <input "
             << "file> <output file>\n";
        exit(1);
    }
"""
# check CLI args are used correctly
if len(sys.argv) != 3:
    print("Usage: python3 personalSales.py <input file> <output file>")
    exit(1)

# --- INPUT + OUTPUT ----------------------------
"""
    personalSales person = loadStruct(argv[1]);

    dump(argv[1],argv[2]);

    return 0;
}
"""
# initialize variables
person = loadStruct(sys.argv[1])

# output
dump(sys.argv[1],sys.argv[2])

