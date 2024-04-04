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
"""
#include <iostream>     // in/out
#include <cstring>      // strcmp()
#include <string>       // atoi(), to_string()
#include <fstream>      // writing to files
#include <cstdlib>      // popen(), pclose()
using namespace std;

bool DEBUG = false;
char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
"""
import sys         # argv
import subprocess  # executing bash commands
import random      # random number gen
import shutil      # moving files

# boolean for debug output (detailed execution)
DEBUG = False
MATRIX = False
# charcter bank for strings
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# --- FUNCTIONS -------------------------------------------------------------------------
# --- EXECUTE COMMAND ---------------------------
"""
int executeCommand(const string&);
int executeCommand(const string& command)
{
    if(DEBUG)
        cout << "\nbeginning method: executeCommand\n";
    
    FILE* pipe = popen(command.c_str(), "r");

    if(!pipe) {
        cerr << "failed to open pipe (popen() failed)).\n";
        exit(3);
    }

    int result; // variable for command result

    if(fscanf(pipe, "%d", &result) != 1) {
        cerr << "failed to read integer from command output.\n";
        exit(4);
    }

    pclose(pipe);

    if(DEBUG)
        cout << "\nend method: executeCommand\n";

    return result;
}
"""
# the function takes a single argument (the shell command) and opens a pipe to execute said command
# pre-condition: the command must be an initialized string in main
# 
# post-condition: the function has opened a pipe, executed a command, and returns the output of
#                 said command, which should be an int
def executeCommand(test_lib) -> int:
    if DEBUG:
        print("\nbeginning method: executeCommand\n")
        
    try:
        # execute the bash command
        result = subprocess.check_output(["ls",test_lib, "|", "wc", "-l"], universal_newlines=True, stderr=subprocess.PIPE, shell=True)
        
        # convert to int
        #count = int(result.strip())
        
        if DEBUG:
            print("\nend method: executeCommand\n")
        
        #return count
        return int(result.strip())
    
    # handle errors if any
    except subprocess.CalledProcessError as e:
        print("Error executing command:", e)
        
        if DEBUG:
            print("\nend method: executeCommand\n")
            
        return None

# --- LOAD MATRIX -------------------------------
"""
void loadMatrix(const int&, const int&, const char*);
void loadMatrix(const int& N, const int& T, const char* datatype)
{
    if(DEBUG)
        cout << "\nbeginning method: loadMatrix\n";
    
    int M;
    cout << "enter second matrix dimension: ";
    cin >> M;

    if(M <= 0) {
        cerr << "error: matrix dimension must be > 0 (provided size: " << M << ").\n";
        exit(2);
    }
    
    cout << "\nYou have chosen to construct:\n" << "matrix: " << N << "x" << M;

    char ident;
    cout << "\nWould you like this matrix to be the identity? [Y/n]: ";
    cin >> ident;

    bool isIdentity = true;
    switch(ident) {
        case 'n':
            isIdentity = false;
            cout << "\nmax value: " << T;
            cout << "\ntype: " << datatype;
            break;
    }
    
    cout << "\nConfirm [Y/n]: ";
    char conf;
    cin >> conf;

    if(conf == 'Y') {
        if(DEBUG)
            cout << "\nloadMatrix: beginning file write\n";

        int testFileNum = executeCommand("ls TestFiles/matrix-test* | wc -l")+1;
        ofstream outputFile ("matrix-test"+to_string(testFileNum));
        
        if(!outputFile) {
            cerr << "Failed to create output file (name used: matrix-test"
                 << testFileNum << ")." << endl;
            exit(5);
        }

        if(N == M)
            outputFile << N << endl;
        else
            outputFile << N << " " << M << endl;

        if(isIdentity) {
            if(N==M) {
                for(int i=0; i<N; i++) {
                    for(int j=0; j<M; j++)
                        if(i == j)
                            outputFile << 1 << " ";
                        else
                            outputFile << 0 << " ";
                    outputFile << endl;
                }
            }
            else {
                cerr << "error: identity matrices must be equal (matrix must be square)\n";
                exit(2);
            }
        }

        if(strcmp(datatype,"INT") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << rand()%T << " ";
                outputFile << endl;
            }
        else if(strcmp(datatype,"DOUBLE") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << (double)(rand()%T)/(double)(rand()%T) << " "; // REFINE THIS
                outputFile << endl;
            }
        else if(strcmp(datatype,"FLOAT") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << (float)(rand()%T)/(float)(rand()%T) << " "; // REFINE THIS
                outputFile << endl;
            }
        else if(strcmp(datatype,"CHAR") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << alphabet[rand()%sizeof(alphabet)] << " ";
                outputFile << endl;
            }
        else {
            cerr << "error: not a valid type for this program, must be INT, "
                 << "DOUBLE, FLOAT, CHAR, or STRING" << endl;
            exit(6);
        }

        outputFile.close();

        string moveFile = "mv matrix-test"+to_string(testFileNum)+" TestFiles";
        system(moveFile.c_str());

        if(DEBUG)
            cout << "\nloadMatrix: end file write\n";
    }
    else if(conf == 'n') {
        cout << "You have chosen to quit the program. Quitting...\n\n";
        exit(0); // not quite accurate
    }
    else {
        cout << "input not valid, respond with 'Y' or 'n': ";
        cin >> conf;
    }

    if(DEBUG)
        cout << "\nend method: loadMatrix\n";
}
"""
# this function creates a file and writes to it the matrix dimension N. If the matrix is not square
# (i.e. N != M), then M is also output. The next N lines are M <numerical datatype>s or characters
# separated by a space. 
# 
# pre-condition: N and T must be defined and initialized to an integer value, and datatype must be
#                initialized to a string or char*
#
# post-condition: nothing in main has changed. the function created an output file and wrote to it
#                 the matrix dimensions and values
def loadMatrix(n: int, t: int, datatype: str):
    if DEBUG:
        print("\nbeginning method: loadMatrix\n")
        
    # prompt for M
    m = int(input("enter second matrix dimension: "))
    # CHECK M
    if m <= 0:
        sys.stderr.write(f"error: matrix dimension must be > 0 (provided size: {m}).\n")
        exit(2)
    
    # CONFIRMATION
    print(f"\nYou have chosen to construct:\nmatrix: {n}x{m}")
    
    # check if user wants to create an identity matrix
    ident = input("\nWould you like this matrix to be the identity? [Y/n]: ")
    # assume true, will override if false
    is_identity = True
    if ident == 'n':
        is_identity = False
        print("\nmax value:",t)
        print("\ntype:",datatype)
    
    conf = input("\nConfirm? [Y/n]: ")
    # check confirmation
    while conf != 'Y' and conf != 'n':
        conf = input("error: please provide [Y/n]: ")
    
    # --- MAIN LOOP -----------------------------
    if conf == 'Y':
        # more detailed documentation found in loadFile function, the approach is the same here but
        # altered to be in matrix form
        
        if DEBUG:
            print("\nloadMatrix: beginning file write\n")
            
        test_file_num = executeCommand("../TestFiles/matrix-test*")+1
        with open(f"matrix-test{test_file_num}", 'w') as file:
            # if the file cannot be created, terminate
            try:
                # write number of cases N to file if N = M, then the matrix is square, and only one
                # dimension will be output. otherwise, output both dimensions
                if n == m:
                    file.write(f"{n}\n")
                else:
                    file.write(f"{n} {m}\n")
                    
                # create identity
                if is_identity:
                    if n==m:
                        for i in range(n):
                            for j in range(m):
                                if i==j:
                                    file.write("1 ")
                                else:
                                    file.write("0 ")
                            file.write("\n")
                    else:
                        sys.stderr.write("error: identity matrices must be equal (matrix must be square).\n")
                        exit(2)
                
                # integer gen
                if datatype == "INT" or datatype == "int":
                    for _ in range(n):
                        for _ in range(m):
                            file.write(str(random.randint(0,t-1)) + " ")
                        file.write("\n")
                # float gen
                elif datatype == "FLOAT" or datatype == 'float':
                    for _ in range(n):
                        for _ in range(m):
                            file.write(str(random.uniform(0,t)) + " ")
                        file.write("\n")
                # char gen
                elif datatype == "CHAR" or datatype == "char":
                    for _ in range(n):
                        for _ in range(m):
                            file.write(random.choice(ALPHABET) + " ")
                        file.write("\n")
                # string gen
                elif datatype == "STRING" or datatype == "string":
                    # define path to 'word bank'
                    dictionary_path = "../Dictionaries/words-alpha.txt"
                    
                    # read dictionary
                    with open(dictionary_path, "r") as dict_file:
                        possible_strings = [line.strip() for line in dict_file]
                    
                    for _ in range(n):
                        for _ in range(m):
                            file.write(random.choice(possible_strings) + " ")
                        file.write("\n")
                # user did not specify a valid data type (for this program)
                else:
                    sys.stderr.write("error: not a valid type for this program, must be INT, DOUBLE, FLOAT, CHAR, or STRING")
                    exit(5)
            
            except IOError:
                sys.stderr.write(f"Failed to create output file (name used: matrix-test{test_file_num}).")
                exit(4)
    
    else:
        print("You have chosen to not load the matrix. Quitting...\n")
        exit(0) # not quite accurate
        
    # move file to appropriate directory -- out of main logic to avoid moving the file while it is
    # open
    shutil.move(f"matrix-test{test_file_num}", "../TestFiles")
        
    if DEBUG:
        print("\nend method: loadMatrix\n")

# --- LOAD FILE ---------------------------------
"""
void loadFile(const int&, const int&, const char*);
void loadFile(const int& N, const int& T, const char* datatype)
{
    if(DEBUG)
        cout << "\nbeginning method: loadFile\n";
    
    cout << "\nYou have chosen:\n" << "number of values: " << N;
    cout << "\nmax value: " << T;
    cout << "\n\nConfirm [Y/n]: ";
    char conf;
    cin >> conf;
    
    if(conf == 'Y') {
        int testFileNum = executeCommand("ls TestFiles/test* | wc -l")+1;

        ofstream outputFile ("test"+to_string(testFileNum));

        if(!outputFile) {
            cerr << "Failed to create output file (name used: test"
                 << testFileNum << ")." << endl;
            exit(5);
        }

        outputFile << N << endl;

        if(strcmp(datatype,"INT") == 0)
            for(int i=0; i<N; i++)
                outputFile << rand()%T << " "; // change " " to \n or endl if  
        else if(strcmp(datatype,"DOUBLE") == 0)// by newline, may update later
            for(int i=0; i<N; i++)
                outputFile << (double)(rand()%T)/(double)(rand()%T) << " "; // REFINE THIS
        else if(strcmp(datatype,"FLOAT") == 0)
            for(int i=0; i<N; i++)
                outputFile << (float)(rand()%T)/(float)(rand()%T) << " "; // REFINE THIS
        else if(strcmp(datatype,"CHAR") == 0)
            for(int i=0; i<N; i++)
                outputFile << alphabet[rand()%sizeof(alphabet)] << " ";
        else if(strcmp(datatype,"STRING") == 0) {
            if(T == -1) {
                for(int i=0; i<N; i++) {
                    for(size_t j=0; j<rand()%sizeof(alphabet); j++)
                        outputFile << alphabet[rand()%sizeof(alphabet)];
                    outputFile << endl;
                }
            }
            else {
                for(int i=0; i<N; i++) {
                    for(int j=0; j<T; j++)
                        outputFile << alphabet[rand()%sizeof(alphabet)];
                    outputFile << endl;
                }
            }
        }
        else {
            cerr << "error: not a valid type for this program, must be INT, "
                 << "DOUBLE, or FLOAT" << endl;
            exit(6);
        }
        
        outputFile.close();

        string moveFile = "mv test"+to_string(testFileNum)+" TestFiles";
        system(moveFile.c_str());
    }
    else if(conf == 'n') {
        cout << "You have chosen to quit the program. Quitting...\n\n";
        exit(0); // not quite accurate
    }
    else {
        cout << "input not valid, respond with 'Y' or 'n': ";
        cin >> conf;
    }

    if(DEBUG)
        cout << "\nend method: loadFile\n";
}
"""
# this function creates a file and first writes to it the amount of data in the file N. The next
# line consists N <numerical datatype>s or characters separated by a space. Strings are generated
# at a random length and separated by a new line.
# pre-condition: N and T must be defined and initialized to an integer value, and datatype must be
#                initialized to a string or char*
# 
# post-condition: nothing in main has changed. the function created an output file and wrote to it
#                 the amount of data and each data value
def loadFile(n,t,datatype):
    if DEBUG:
        print("\nbeginning method: loadFile\n")
        
    # confirmation
    print("\nYou have chosen:\nnumber of values:",n,"\nmax value:",t)
    conf = input("\n\nConfirm [Y/n]: ")
    # check confirmation
    while conf != 'Y' and conf != 'n':
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
        test_file_num = executeCommand("../TestFiles/test*")+1
        # check if none for exit
        if type(test_file_num) == None:
            print("error: unable to progress without command output")
            exit(3)
        
        try:
            with open("test"+str(test_file_num)) as file:
                # write number of cases to file
                file.write(n,"\n")
                
                # integer
                if datatype == "INT" or datatype == "int":
                    for _ in range(n):
                        file.write(str(random.randint(0,t-1)) + " ") # change " " to \n if data 
                # float                                              # needs to be separated by
                elif datatype == "FLOAT" or datatype == "float":     # newline, may update later
                    for _ in range(n):
                        file.write(str(random.uniform(0,t)) + " ")
                # char
                elif datatype == "CHAR" or datatype == "char":
                    for _ in range(n):
                        file.write(str(random.choice(ALPHABET)) + " ")
                # string
                elif datatype == "STRING" or datatype == "string":
                    # word bank
                    if t == -1:
                        # define path to 'word bank'
                        dictionary_path = "../Dictionaries/words-alpha.txt"
                        
                        # read dictionary
                        with open(dictionary_path, "r") as dict_file:
                            possible_strings = [line.strip() for line in dict_file]
                        
                        for _ in range(n):
                            file.write(random.choice(possible_strings) + " ")
                    # uniform string length
                    elif t == -2:
                        t = int(input("Enter the length of the strings you want: "))
                        
                        # print n words
                        for _ in range(n):
                            # get random char
                            for _ in range(t):
                                file.write(ALPHABET[random.randint(len(ALPHABET))] + " ")
                    # variable string length
                    else:
                        # print n words
                        for _ in range(n):
                            # get random char
                            for _ in range(random.randint(len(ALPHABET))):
                                file.write(ALPHABET[random.randint(len(ALPHABET))] + " ")
                
                else:
                    sys.stderr.write("error: not a valid type for this program, must be INT, DOUBLE, or FLOAT\n")
                    exit(5)
            
        except IOError:
            sys.stderr.write(f"Failed to create output file (name used: test{test_file_num}).\n")
            exit(4)
        
    else:
        print("You have chosen to not load the file. Quitting...\n")
        exit(0) # not quite accurate
    
    # use CLI to move generated file to appropriate directory
    shutil.move(f"test{test_file_num}", "../TestFiles")
    # could remove this if directory included in file creation command
        
    if DEBUG:
        print("\nend method: loadFile\n")

# --- MAIN ------------------------------------------------------------------------------
# --- CHECK CLI ARGS ----------------------------
"""
int main(int argc, char* argv[])
{
    if(argc < 4 || argc > 6) {
        cerr << "error: must have 4, 5, or 6 arguments: exe, -d flag (optional)"
             << ", -m flag (optional), number of cases, range of values, "
             << "datatype. only " << argc << " arguments were provided." << endl;
        return 1; // return 1 (stdout) vs. return 2 (stderr)?
    }
"""
# check if I/O redirection is used correctly (must be 4, 5, or 6 flags) 4 required flags, +2
# optional (6)
if len(sys.argv) < 4 or len(sys.argv) > 6:
    sys.stderr.write(f"""error: invalid arguments, {len(sys.argv)} provided.
                     Usage: python3 create_test-<version>.py [-d] [-m] <number of cases> <range> <type>""")

# --- INTRODUCTION ------------------------------
"""
    cout << "This program generates a test file (./TestFiles) where the "
         << "user specifies the number of values, range, and type\n";
"""
print("""This program generates a test file (./TestFiles) where the user specifies the number of
      values, range, and type\n""")

# --- CONFIRMATION ------------------------------
"""
    cout << "Do you want to run this program? [Y/n]: ";
    char confirmation;
    cin >> confirmation;

    if(confirmation == 'n') {
        cout << "terminating...\n";
        exit(0);
    }
"""
confirmation = input("Do you want to run this program? [Y/n]: ")
# if declined, terminate
if confirmation == 'n':
    print("terminating...\n")
    exit(0)
    # using 0 for exit because it is successful - user specified would normally use >0 if for error

# --- VAR SETUP ---------------------------------
"""
    int N, T;
"""
# number of cases, range of values created here because if -m is present, then the values in argv
# shift
t: int
n: int

# --- LOGIC BASED ON CLI ARGS -----------------------------------------------------------
# --- DEBUG FLAG --------------------------------
"""
    if(strcmp(argv[1],"-d") == 0 || strcmp(argv[2],"-d") == 0) {
        DEBUG = true;

        if(strcmp(argv[1],"-m") == 0 || strcmp(argv[2],"-m") == 0) {

            N = atoi(argv[3]);
            T = atoi(argv[4]);

            if(N <= 0) {
                cerr << "error: matrix dimension must be > 0 (provided size: " << N << ").\n";
                exit(2);
            }

            if(T <= 0 && strcmp(argv[5],"CHAR") != 0) {
                cerr << "error: range of values must be > 0 (provided length: " << T << ").\n";
                exit(2);
            }

            loadMatrix(N,T,argv[5]);
        }
        """
# if -d specified, enable debug output first comparison is how flags should be ordered, second is
# contingency for if the user swaps -d and -m
if sys.argv[1] == '-d' or sys.argv[2] == '-d':
    # d IS PRESENT
    DEBUG = True
    
    # same contingency here
    if sys.argv[1] == '-m' or sys.argv[2] == '-m':
        # BOTH -d AND -m FLAGS ARE PRESENT: N = argv[3]
        MATRIX = True
        
        # convert CLI arguments to ints, atoi (notes)
        n = int(sys.argv[3])
        t = int(sys.argv[4])
        
        # CHECK N
        if n <= 0:
            sys.stderr.write(f"error: matrix dimension must be > 0 (provided size: {n}).\n")
            exit(2)
        # CHECK T
        # range of numerical values, so must be > 0
        # can ignore whatever T is if we are processing characters
        if t <= 0 and sys.argv[5] != "CHAR":
            sys.stderr.write(f"error: range of values must be > 0 (provided length: {t}).\n")
            exit(2)
        
        # construct matrix test file
        loadMatrix(n,t,sys.argv[5])
        
# ---  --------------------------------
        """
        else {
            N = atoi(argv[2]);
            T = atoi(argv[3]);

            if(N <= 0) {
                cerr << "error: amount of data must be > 0 (provided length: " << N << ").\n";
                exit(2);
            }

            if(strcmp(argv[4],"CHAR") != 0)
                if(T == 0 || T < -1) {
                    cerr << "error: range of values must not equal 0 or be less "
                         << "than -1 (provided length: " << T << ").\n";
                    exit(2);
                }

            loadFile(N,T,argv[4]);
        }
    }
    """
    else:
        # ONLY -d IS PRESENT: N = argv[2]
        n = int(sys.argv[2])
        t = int(sys.argv[3])
        
        # CHECK N
        if n <= 0:
            sys.stderr.write(f"error: matrix dimension must be > 0 (provided length: {n}).\n")
            exit(2)
        # CHECK T
        # if the type is not a char, then ignore (char does not need length)
        if sys.argv[4] != "CHAR":
            # includes strings, so special case -1 must pass check
            if t == 0 or t < -1:
                sys.stderr.write(f"error: range of values must not equal 0 or be less than -1 (provided length: {t}).\n")
                exit(2)
        
        # load test file
        loadFile(n,t,sys.argv[4])
    
# ---  --------------------------------
    """
    else {
        if(strcmp(argv[1],"-m") == 0) {
            N = atoi(argv[2]);
            T = atoi(argv[3]);

            if(N <= 0) {
                cerr << "error: matrix dimension must be > 0 (provided size: " << N << ").\n";
                exit(2);
            }
        
            if(T <= 0 && strcmp(argv[4],"CHAR") != 0) {
                cerr << "error: range of values must be > 0 (provided length: " << T << ").\n";
                exit(2);
            }

            loadMatrix(N,T,argv[4]);
        }
        """
else:
    # d IS NOT PRESENT
    
    # since -d is not present, only one flag to check
    if sys.argv[1] == 'm':
        # ONLY -m IS PRESENT: N = argv[2]
        MATRIX = True
        
        n = int(sys.argv[2])
        t = int(sys.argv[3])
    
        # CHECK N
        if n <= 0:
            sys.stderr.write(f"error: matrix dimension must be > 0 (provided size: {n}).\n")
            exit(2)
        # CHECK T
        # range of numerical values, so must be > 0
        # can ignore whatever T is if we are processing characters
        if t <= 0 and sys.argv[5] != "CHAR":
            sys.stderr.write(f"error: range of values must be > 0 (provided length: {t}).\n")
            exit(2)
        
        # construct matrix test file
        loadMatrix(n,t,sys.argv[4])
        
# ---  -------------------------------- 
        """
        else {
            N = atoi(argv[1]);
            T = atoi(argv[2]);

            if(N <= 0) {
                cerr << "error: amount of data must be > 0 (provided length: " << N << ").\n";
                exit(2);
            }

            if(strcmp(argv[3],"CHAR") != 0)
                if(T == 0 || T < -1) {
                    cerr << "error: range of values must not equal 0 or be less "
                         << "than -1 (provided length: " << T << ").\n";
                    exit(2);
                }

            loadFile(N,T,argv[3]);
        }
    }
    
    return 0;
}
    """
    else:
        n = int(sys.argv[1])
        t = int(sys.argv[2])
        
        # CHECK N
        if n <= 0:
            sys.stderr.write(f"error: matrix dimension must be > 0 (provided length: {n}).\n")
            exit(2)
        # CHECK T
        # if the type is not a char, then ignore (char does not need length)
        if sys.argv[4] != "CHAR":
            # includes strings, so special case -1 must pass check
            if t == 0 or t < -1:
                sys.stderr.write(f"error: range of values must not equal 0 or be less than -1 (provided length: {t}).\n")
                exit(2)

        # load test file
        loadFile(n,t,sys.argv[3])



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


# BACKGROUND ON BASH COMMANDS IN THIS PROGRAM
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
# to_string() -> taken from ChatGPT, researched at:
# https://en.cppreference.com/w/cpp/string/basic_string/to_stringhowever
# 
# 
# implementation of bash commands in a program -> taken from ChatGPT
# 
# 
# general pipe understanding:
# https://www.man7.org/linux/man-pages/man2/pipe.2.html
# https://stackoverflow.com/questions/4812891/fork-and-pipes-in-c
# 
# 
# c_str():
# "Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a
# C-string) representing the current value of the string object." In other words, convert to a
# string.
# https://cplusplus.com/reference/string/string/c_str/
# 
# 
# fscanf return value:
# "On success, the function returns the number of items of the argument list successfully filled.
# This count can match the expected number of items or be less (even zero) due to a matching
# failure, a reading error, or the reach of the end-of-file." 
# We want the function to return 1 item: the number of files. Therefore, the 'number of items
# successfully filled' should be 1. So if the function returns a value other than that (1),
# something went wrong.
# https://cplusplus.com/reference/cstdio/fscanf/
# 
#
# convert CLI arguments to ints (atoi, stoi) -> from ChatGPT
# 
# 
# atoi -> ASCII to integer
# stoi -> string to integer
# https://stackoverflow.com/questions/37838417/what-do-atoi-atol-and-stoi-stand-for
# 
# 
# random string generation:
# https://stackoverflow.com/questions/47977829/generate-a-random-string-in-c11
# 
# 
# size_t -> from ChatGPT
# "The warning occurs because the type of word.length() is size_t, which is an unsigned integer
# type, and the loop variable i is of type int, which is signed"
 
