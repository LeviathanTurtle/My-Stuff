# CREATE A TEST FILE OF INTEGERS, FLOATS, CHARACTERS, OR STRINGS -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.5.2023
# Doctored: 10.7.2023*
# Python-ized: 4.2.2024
# 
# [DESCRIPTION]:
# This program generates a test file of random datatypes (stored in TestFiles of current directory)
# where the user specifies the number of values [number of cases], range of values used [range],
# and the datatype [type]. Valid data types are: integers, doubles, floats, characters, and
# strings. All data types can be used with matrix construction except strings. Using these values,
# the program will create and write to a file. The size of the data N (how many values in the file)
# is first number and is separated by a new line, while the next N values (data) are separated by a
# space. If the optional -m flag is present, the program will output the data in matrix form: first
# the dimensions, then the data. Matrix output is limited to numerical datatypes and characters.
# Revision hitory and notes are at the bottom.
# 
# [USAGE]:
# python3 create_test-<version>.py [-d] [-m] <number of cases> <range> <type>
# 
# [-d]              - optional, enable debug options (detailed execution)
# [-m]              - optional, specify if program is to generate matrices
# <number of cases> - how many entries in the file
# <range>           - range of values to be used
# <type>            - can be: INT, DOUBLE, FLOAT, CHAR, STRING
# 
# [NOTES]:
# A few (notes) are in comments. This is to show that I had to research something or ask ChatGPT.
# The topics with associated links where I found the information are below the code.
#  
# Also, this program assumes any existing test files are properly numbered based on the number of
# files in the directory (test1, test2, etc.). It WILL append to a test file if, for example, there
# are 2 test files, but are named 'testX' and 'test3'.
# 
# For generating strings, if you want to use random words from a 'word bank,' use the value -1 for
# the range. If you want a uniform string length, use the value -2 for the range. Otherwise, it
# will generate a string of characters of the length you specify for the range. For character
# generation, T is ignored, so input any integer value.
# 
# [EXIT/TERMINATING CODES]:
# 0 - the user specified 'n' when prompted if information was correct or the
#     program successfully ran through a complete execution
# 
# 1 - execution arguments were used incorrectly
# 
# 2 - invalid matrix dimension(s), number of values, or range of values
# 
# 3 - failed to read integer from command output
# 
# 4 - file failed to be opened or created
# 
# 5 - invalid data type was used

# --- IMPORTS + GLOBAL VARS -------------------------------------------------------------
from sys import argv, stderr, exit
from subprocess import check_output, CalledProcessError, PIPE
from random import randint, uniform, choice
from shutil import move
from typing import Optional

# boolean for debug output (detailed execution)
DEBUG: bool = False
MATRIX: bool = False
# charcter bank for strings
ALPHABET: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- FUNCTIONS -------------------------------------------------------------------------
# --- EXECUTE COMMAND ---------------------------
# pre-condition: test_lib must be a valid directory path as a string
# post-condition: the function returns the count of files in the directory as an integer or None if
#                 an error occurs during command execution
def execute_command(test_lib: str) -> Optional[int]:
    """Execute a shell command to count the number of files in a directory."""
    
    if DEBUG:
        print("Entering execute_command...")
        
    try:
        # execute the bash command
        result: str = check_output(
            f"ls {test_lib} | wc -l", 
            universal_newlines=True, 
            stderr=PIPE, 
            shell=True
        )
        
        if DEBUG:
            print("Exiting execute_command.")
        return int(result.strip())
    
    # handle errors if any
    except CalledProcessError as e:
        print("Error executing command:", e)
        
        if DEBUG:
            print("Exiting execute_command.")
        return None


# --- LOAD MATRIX -------------------------------
# pre-condition: N and T must be a positive integer value, and datatype must be of type INT, FLOAT,
#                or STRING
# post-condition: a matrix file is created and moved to the "../TestFiles" directory, or the
#                 program exits if invalid input or errors occur during file creation
def load_matrix(n: int, t: int, datatype: str) -> None:
    """Load a matrix with specified dimensions and data type, and save it to a file."""
    
    if DEBUG:
        print("Entering load_matrix...")
        
    # prompt for M
    m = int(input("enter second matrix dimension: "))
    # CHECK M
    if m <= 0:
        stderr.write(f"error: matrix dimension must be > 0 (provided size: {m}).\n")
        exit(2)
    
    # CONFIRMATION
    print(f"\nYou have chosen to construct:\nmatrix: {n}x{m}")
    
    # check if user wants to create an identity matrix
    is_identity: bool = input("\nWould you like this matrix to be the identity? [Y/n]: ") != 'n'
    
    print("\nmax value: ",t)
    print("\ntype: ",datatype)
    
    conf = input("\nConfirm? [Y/n]: ")
    # check confirmation
    while conf not in ('Y', 'n'):
        conf = input("error: please provide [Y/n]: ")
    
    # --- MAIN LOOP -----------------------------
    if conf == 'Y':
        # more detailed documentation found in loadFile function, the approach is the same here but
        # altered to be in matrix form
        
        if DEBUG:
            print("\nloadMatrix: beginning file write\n")
            
        test_file_num = execute_command("../TestFiles/matrix-test*")+1
        file_name: str = f"matrix-test{test_file_num}"
        
        try:
            with open(file_name, 'w') as file:
                # write number of cases N to file if N = M, then the matrix is square, and only one
                # dimension will be output. otherwise, output both dimensions
                if n == m:
                    file.write(f"{n}\n")
                else:
                    file.write(f"{n} {m}\n")
                    
                # create identity
                if is_identity and n == m:
                    for i in range(n):
                        for j in range(m):
                            file.write("1 " if i == j else "0 ")
                        file.write("\n")
                elif is_identity:
                    stderr.write("Error: Identity matrices must be square.\n")
                    exit(2)
                else:
                    _write_matrix_data(file, n, m, t, datatype)
        except IOError:
            stderr.write(f"Failed to create output file (name used: {file_name}).\n")
            exit(4)
        
        # move file to appropriate directory -- out of main logic to avoid moving the file while it
        # is open
        move(file_name, "../TestFiles")     
    else:
        print("You have chosen to not load the matrix. Quitting...\n")
        if DEBUG:
            print("Exiting load_matrix.")
        exit(0) # not quite accurate
        
    if DEBUG:
        print("Exiting load_matrix.")


# --- WRITE MATRIX DATA -------------------------
# pre-condition: file is an open file object, n, m, t are integers, datatype is of type INT, FLOAT,
#                or STRING
# post-condition: matrix data is written to the file
def _write_matrix_data(file, n: int, m: int, t: int, datatype: str) -> None:
    """Helper function to write matrix data to a file based on the datatype."""
    
    if DEBUG:
        print("Entering _write_matrix_data...")
    
    if datatype.lower() == "int":
        for _ in range(n):
            for _ in range(m):
                file.write(f"{randint(0, t-1)} ")
            file.write("\n")
    elif datatype.lower() == "float":
        for _ in range(n):
            for _ in range(m):
                file.write(f"{uniform(0, t):.6f} ")
            file.write("\n")
    elif datatype.lower() == "char":
        for _ in range(n):
            for _ in range(m):
                file.write(choice(ALPHABET) + " ")
            file.write("\n")
    elif datatype.lower() == "string":
        dictionary_path: str = "../Dictionaries/words-alpha.txt"
        with open(dictionary_path, "r") as dict_file:
            possible_strings = [line.strip() for line in dict_file]
        for _ in range(n):
            for _ in range(m):
                file.write(choice(possible_strings) + " ")
            file.write("\n")
    else:
        stderr.write("Error: Invalid datatype for this program. Must be INT, FLOAT, CHAR, or STRING.\n")
        exit(5)

    if DEBUG:
        print("Exiting _write_matrix_data.")


# --- LOAD FILE ---------------------------------
# pre-condition: n is a positive integer, t is an integer specifying the range/length of values,
#                datatype is of type INT, FLOAT, or STRING
# post-condition: a data file is created and moved to the "../TestFiles" directory, or the program
#                 exits if invalid input or errors occur during file creation.
def load_file(n: int, t: int, datatype: str) -> None:
    """Load a file with specified number of values and data type, and save it to a file."""
    
    if DEBUG:
        print("Entering load_file...")
        
    # confirmation
    print("\nYou have chosen:\nnumber of values:",n,"\nmax value:",t)
    conf = input("\n\nConfirm [Y/n]: ")
    # check confirmation
    while conf not in ('Y', 'n'):
        conf = input("error: please provide [Y/n]: ")
    
    # this is what the user wants to do
    if conf == 'Y':
        # create a file for output with corresponding test number

        # the CLI command in the executeCommand() function will return the number of files in
        # "./TestFiles". This number will be used to create the next test file with the
        # corresponding number, which is why we add 1 since we do not want to create a file with a
        # duplicate name. 

        # For example, if the command returns 2 (meaning 2 files are in the TestFiles directory),
        # then a file will be created named "test3".
        test_file_num = execute_command("../TestFiles/test*")+1
        # check if none for exit
        if test_file_num is None:
            print("error: unable to progress without command output")
            exit(3)
        
        file_name: str = f"test{test_file_num}"
        
        try:
            with open(file_name, 'w') as file:
                # write number of cases to file
                file.write(f"{n}\n")
                _write_file_data(file, n, t, datatype)
        except IOError:
            stderr.write(f"Failed to create output file (name used: test{test_file_num}).\n")
            exit(4)
        
    else:
        print("You have chosen to not load the file. Quitting...\n")
        exit(0) # not quite accurate
    
    # use CLI to move generated file to appropriate directory
    move(f"test{test_file_num}", "../TestFiles")
    # could remove this if directory included in file creation command
        
    if DEBUG:
        print("Exiting load_file...")


# --- WRITE FILE DATA ---------------------------
# pre-condition: file is an open file object, n is an integer representing the number of values, t
#                is an integer representing the range or length of values, datatype is one of INT,
#                FLOAT, or STRING
# post-condition: data is written to the file
def _write_file_data(file, n: int, t: int, datatype: str) -> None:
    """Helper function to write data to a file based on the datatype."""
    
    if DEBUG:
        print("Entering _write_file_data...")
    
    if datatype.lower() == "int":
        for _ in range(n):
            file.write(f"{randint(0, t-1)} ")
    elif datatype.lower() == "float":
        for _ in range(n):
            file.write(f"{uniform(0, t):.6f} ")
    elif datatype.lower() == "char":
        for _ in range(n):
            file.write(choice(ALPHABET) + " ")
    elif datatype.lower() == "string":
        dictionary_path: str = "../Dictionaries/words-alpha.txt"
        with open(dictionary_path, "r") as dict_file:
            possible_strings = [line.strip() for line in dict_file]
        for _ in range(n):
            file.write(choice(possible_strings) + " ")
    else:
        stderr.write("Error: Invalid datatype for this program. Must be INT, FLOAT, CHAR, or STRING.\n")
        exit(5)
    
    if DEBUG:
        print("Exiting _write_file_data.")


def main():
    # --- CHECK CLI ARGS ------------------------
    # check if I/O redirection is used correctly (must be 4, 5, or 6 flags) 4 required flags, +2
    # optional (6)
    if len(argv) < 4 or len(argv) > 6:
        stderr.write(f"""error: invalid arguments, {len(argv)} provided.
        Usage: python3 create_test-<version>.py [-d] [-m] <number of cases> <range> <type>""")
        exit(1)

    # --- INTRODUCTION --------------------------
    print("""This program generates a test file (./TestFiles) where the user specifies the number
          of values, range, and type\n""")

    # --- CONFIRMATION --------------------------
    confirmation = input("Do you want to run this program? [Y/n]: ")
    # if declined, terminate
    if confirmation == 'n':
        print("terminating...\n")
        exit(0)
        # using 0 for exit because it is successful - user specified would normally use >0 if for
        # error

    # --- VAR SETUP -----------------------------
    # number of cases, range of values created here because if -m is present, then the values in
    # argv shift
    t: int
    n: int

    # --- DEBUG FLAG --------------------------------
    # if -d specified, enable debug output first comparison is how flags should be ordered, second
    # is contingency for if the user swaps -d and -m
    if '-d' in argv:
        DEBUG = True

    if '-m' in argv:
        MATRIX = True
    
    n = int(argv[-3])
    t = int(argv[-2])
    datatype: str = argv[-1]

    # check dimension args
    if n <= 0:
        stderr.write(f"Error: Dimension must be > 0 (provided size: {n}).\n")
        exit(2)

    # check numerical range and string length args
    if t <= 0 and datatype.lower() != "char":
        stderr.write(f"Error: Range of values must be > 0 (provided length: {t}).\n")
        exit(2)

    if MATRIX:
        load_matrix(n, t, datatype)
    else:
        load_file(n, t, datatype)


if __name__ == "__main__":
    main()


# REVISION HISTORY
# Updated: 10.23.2023 -- added optional flag -m to specify matrix construction, offloaded bulk of
#                        program to functions (made it more modular), updated documentation, added
#                        function prototypes, updated confirmation checks, included filename in
#                        error for file creation checks, updated argument count error to show a
#                        more helpful message.
# 
# Updated: 10.24.2023 -- added random character and string generation, added optional debug flag
#                        -d, various output messages if in debugging mode, updated documentation,
#                        added range of values check.
#
# Updated: 10.31.2023 -- promoted alphabet and DEBUG flag to global variables
# 
# Updated: 3.20.2024 -- added option to create identity matrix. TODO: export a few things to new
#                       functions.
# 
# Updated: 4.03.2024 -- translated to Python and added string generation.
# 
# Updated: 8.17.2024 -- function decomposition and PEP 8 Compliance


# BACKGROUND ON BASH COMMANDS IN THIS PROGRAM -- C++
#
# THE IDEA IS TO EXECUTE A BASH COMMAND TO HELP CREATE THE PROPER TEST FILE NAME
# This was my original approach:
#
#      string filenum = argv[4];   <- argv[4] is the command
#      ofstream outfile ("test"+filenum);
#
# after errors (and noticing it would create the wrong named file), it became:
# 
#      string filenum = argv[4];
#      int number = stoi(filenum)+1;
#      ofstream outfile ("test"+to_string(number));
# 
# after executing, the following error came:
# 
#      terminate called after throwing an instance of 'std::invalid_argument'
#      what(): stoi
#      Aborted (core dumped).
# 
# after some research, I learned this is when stoi() fails to convert string to integer. I asked
# ChatGPT for a solution, and I used what it gave me.


# FULL LIST OF THINGS REFERENCED, RESEARCHED, OR TAKEN FROM CHATGPT
#
# 
# random.stuff
# https://docs.python.org/3/library/random.html
 
