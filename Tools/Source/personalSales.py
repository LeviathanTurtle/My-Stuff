# PERSONAL COFFEE SALES STATISTICS -- V.PY
# William Wadsworth
# CSC1710
# Created: 9.16.2020
# Doctored: 11.2.2023
# 
# Python-ized: 3.30.2024
# Updated 8.17.2024: function decomposition and PEP 8 Compliance
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
from sys import argv, stderr, exit
from dataclasses import dataclass

DEBUG: bool = False


# --- OBJECTS ---------------------------------------------------------------------------
@dataclass
class PersonalSales:
    first_name: str = ""
    last_name: str = ""
    department: str = ""
    salary: float = 0.0
    bonus: float = 0.0
    taxes: float = 0.0
    distance: float = 0.0
    time: float = 0.0
    cups: int = 0
    cup_cost: float = 0.0


# --- FUNCTIONS -------------------------------------------------------------------------
# --- LOAD PERSONAL SALES -----------------------
# pre-condition: input_file is the path to a file containing personal sales data in a specific 
#                format
# post-condition: returns a populated PersonalSales object with data from the file
def load_personal_sales(input_file: str) -> PersonalSales:
    """Loads data from a file into a PersonalSales object."""
    
    if DEBUG:
        print("Entering load_personal_sales...")
        
    person = PersonalSales()
    
    try:
        with open(input_file,'r') as file:            
            # take first and last values from data file, display in output
            person.first_name = file.readline().strip()
            person.last_name = file.readline().strip()
            person.department = file.readline().strip()
            print(f"Name: {person.first_name} {person.last_name}, Department: {person.department}")

            # take salary, bonus, and tax values from data file, display in output, set to display
            # two decimal places
            person.salary = float(file.readline().strip())
            person.bonus = float(file.readline().strip())
            person.taxes = float(file.readline().strip())
            print(f"Monthly Gross Income: ${person.salary:.2f}, Bonus: {person.bonus:.2f}%, Taxes: {person.taxes:.2f}%")

            # take distance and time values from data file, display in output, calculate mph
            person.distance = float(file.readline().strip())
            person.time = float(file.readline().strip())
            print(f"Distance traveled: {person.distance:.2f} miles, Traveling Time: {person.time:.2f} hours")
            print(f"Average Speed: {person.distance/person.time:.2f} miles per hour")

            # take cups and cost values from data file, display in output
            person.cups = int(file.readline().strip())
            person.cup_cost = float(file.readline().strip())
            print(f"Number of coffee cups sold: {person.cups}, Cost: ${person.cup_cost:.2f} per cup")
            print(f"Sales amount = ${person.cups*person.cup_cost:.2f}")
    except IOError:
        stderr.write(f"error: file (name: {input_file}) unable to be opened or created.")
        exit(2)
    
    if DEBUG:
        print("Exiting load_personal_sales.")
    return person    


# --- DUMP FILE ---------------------------------
# pre-condition: input_file_name is the path to the input file, output_file_name is the path to the
#                output file
# post-condition: the contents of the input file are written to the output file
def dump_file(input_file_name: str, output_file_name: str) -> None:
    """Copies the contents of the input file to the output file."""
    
    if DEBUG:
        print("Entering dump_file...")
        
    # check files were opened
    try:
        with open(input_file_name, 'r') as input_file, open(output_file_name, 'w') as output_file:
            # read input file and write its contents to the output file
            output_file.write(input_file.read())
    except IOError:
        print(f"error: file(s) unable to be opened or created (input: {input_file_name}, output: {output_file_name}).")
        exit(2)
    
    if DEBUG:
        print("Exiting dump_file.")


def main():
    # --- CHECK CLI ARGS ----------------------------
    # check CLI args are used correctly
    if len(argv) != 3:
        stderr.write("Usage: python3 personalSales.py <input file> <output file>")
        exit(1)

    # --- INPUT + OUTPUT ----------------------------
    # initialize variables
    person = load_personal_sales(argv[1])

    # output
    print(f"\nPersonal sales: {person}")
    dump_file(argv[1],argv[2])


if __name__ == "__main__":
    main()