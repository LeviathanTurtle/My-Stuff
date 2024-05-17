/* CREATE A TEST FILE OF INTEGERS, DOUBLES, FLOATS, CHARACTERS, OR STRINGS
 * WILLIAM WADSWORTH
 * Created: 10.5.2023
 * Doctored: 10.7.2023*
 * Updated: 10.23.2023 -- added optional flag -m to specify matrix construction,
 *                        offloaded bulk of program to functions (made it more
 *                        modular), updated documentation, added function
 *                        prototypes, updated confirmation checks, included
 *                        filename in error for file creation checks, updated 
 *                        argument count error to show a more helpful message.
 * 
 * Updated: 10.24.2023 -- added random character and string generation, added
 *                        optional debug flag -d, various output messages if in
 *                        debugging mode, updated documentation, added range of
 *                        values check.
 *
 * Updated: 10.31.2023 -- promoted alphabet and DEBUG flag to global variables
 * CSC-4510
 * ~/csc4510/prog3-sort/Source/create_test.cpp
 * 
 * 
 * [DESCRIPTION]:
 * This program generates a test file of random datatypes (stored in TestFiles
 * of current directory) where the user specifies the number of values [number
 * of cases], range of values used [range], and the datatype [type]. Valid data
 * types are: integers, doubles, floats, characters, and strings. All data
 * types can be used with matrix construction except strings. Using these
 * values, the program will create and write to a file. The size of the data N
 * (how many values in the file) is first number and is separated by a new
 * line, while the next N values (data) are separated by a space. If the
 * optional -m flag is present, the program will output the data in matrix
 * form: first the dimensions, then the data. Matrix output is limited to
 * numerical datatypes and characters.
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ create_test.cpp -Wall -o create_test
 *
 * To run (4 min, 6 max args):
 *     ./create_test [-d] [-m] <number of cases> <range> <type> 
 *
 * [-d]              - optional, enable debug options (detailed execution)
 * [-m]              - optional, specify if program is to generate matrices
 * <number of cases> - how many entries in the file
 * <range>           - range of values to be used
 * <type>            - can be: INT, DOUBLE, FLOAT, CHAR, STRING
 * 
 * 
 * [NOTES]:
 * A few (notes) are in comments. This is to show that I had to research 
 * something or ask ChatGPT. The topics with associated links where I found the
 * information are below the code.
 * 
 * Also, this program assumes any existing test files are properly numbered
 * based on the number of files in the directory (test1, test2, etc.). It WILL
 * append to a test file if, for example, there are 2 test files, but are named
 * 'testX' and 'test3'.
 * 
 * For generating strings, if you do not want uniform string length, use the
 * value -1 for the range. Otherwise, it will generate a word of the length you
 * specify for the range. For character generation, T is ignored, so input any
 * integer value. 
 * 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - the user specified 'n' when prompted if information was correct or the
 *     program successfully ran through a complete execution
 * 
 * 1 - execution arguments were used incorrectly
 * 
 * 2 - invalid matrix dimension(s), number of values, or range of values
 * 
 * 3 - pipe failed to open
 * 
 * 4 - failed to read integer from command output
 * 
 * 5 - file failed to opened or created
 * 
 * 6 - invalid data type was used
*/

#include <iostream>     // in/out
#include <cstring>      // strcmp()
#include <string>       // atoi(), to_string()
#include <fstream>      // writing to files
#include <cstdlib>      // popen(), pclose()
using namespace std;

// THE IDEA IS TO EXECUTE A BASH COMMAND TO HELP CREATE THE PROPER TEST FILE NAME
/* THE PROCESS
 * This was my original approach:
 *
 *      string filenum = argv[4];   <- argv[4] is the command
 *      ofstream outfile ("test"+filenum);
 *
 * after errors (and noticing it would create the wrong named file), it became:
 * 
 *      string filenum = argv[4];
 *      int number = stoi(filenum)+1;
 *      ofstream outfile ("test"+to_string(number));
 * 
 * after executing, the following error came:
 * 
 *      terminate called after throwing an instance of 'std::invalid_argument'
 *      what(): stoi
 *      Aborted (core dumped).
 * 
 * after some research, I learned this is when stoi() fails to convert string
 * to integer. I asked ChatGPT for a solution, and this is what it gave me.
*/


// function prototypes
//                  CLI command
int executeCommand(const string&);
//                   N           T        datatype
void loadMatrix(const int&, const int&, const char*);
//                 N           T        datatype
void loadFile(const int&, const int&, const char*);


// global variables
// boolean for debug output (detailed execution)
bool DEBUG = false;
// charcter bank for strings
char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";


// maybe use printf instead of cout?
int main(int argc, char* argv[])
{
    // introduction
    cout << "This program generates a test file (./TestFiles) where the "
         << "user specifies the number of values, range, and type\n";


    // confirm
    cout << "Do you want to run this program? [Y/n]: ";
    char confirmation;
    cin >> confirmation;
    // if declined, terminate
    if(confirmation == 'n') {
        cout << "terminating...\n";
        exit(0);
    }
    // using 0 for exit because it is successful - user specified
    // would normally use >0 if for error

    // check if I/O redirection is used correctly (must be 4, 5, or 6 flags)
    // 4 required flags, +2 optional (6)
    //if(argc != 4 && argc != 5 && argc != 6) {
    if(argc < 4 || argc > 6) {
        cerr << "error: must have 4, 5, or 6 arguments: exe, -d flag (optional)"
             << ", -m flag (optional), number of cases, range of values, "
             << "datatype. only " << argc << " arguments were provided." << endl;
        return 1; // return 1 (stdout) vs. return 2 (stderr)?
    }
    

    // number of cases, range of values
    // created here because if -m is present, then the values in argv shift
    int N, T;
    // boolean for debug output
    //bool DEBUG = false;
    // promote to global variable? would not have to pass to each function



    // should this be cleaned up? can it?

    // if -d specified, enable debug output
    // first comparison is how flags should be ordered, second is contingency
    // for if the user swaps -d and -m
    if(strcmp(argv[1],"-d") == 0 || strcmp(argv[2],"-d") == 0) {
        // -d IS PRESENT
        DEBUG = true;

        // same contingency here
        if(strcmp(argv[1],"-m") == 0 || strcmp(argv[2],"-m") == 0) {
            // BOTH -d AND -m FLAGS ARE PRESENT: N = argv[3]

            // convert CLI arguments to ints, atoi (notes)
            // char* -> int
            N = atoi(argv[3]);
            T = atoi(argv[4]);

            // CHECK N
            if(N <= 0) {
                cerr << "error: matrix dimension must be > 0 (provided size: " << N << ").\n";
                exit(2);
            }
            // CHECK T
            // range of numerical values, so must be > 0
            // can ignore whatever T is if we are processing characters
            if(T <= 0 && strcmp(argv[5],"CHAR") != 0) {
                cerr << "error: range of values must be > 0 (provided length: " << T << ").\n";
                exit(2);
            }

            // construct matrix test file
            loadMatrix(N,T,argv[5]);
        }
        else {
            // ONLY -d IS PRESENT: N = argv[2]
            N = atoi(argv[2]);
            T = atoi(argv[3]);

            // CHECK N
            if(N <= 0) {
                cerr << "error: amount of data must be > 0 (provided length: " << N << ").\n";
                exit(2);
            }
            // CHECK T
            // if the type is not a char, then ignore (char does not need length)
            if(strcmp(argv[4],"CHAR") != 0)
                // includes strings, so special case -1 must pass check
                if(T == 0 || T < -1) {
                    cerr << "error: range of values must not equal 0 or be less "
                         << "than -1 (provided length: " << T << ").\n";
                    exit(2);
                }

            // load test file
            loadFile(N,T,argv[4]);
        }
    }
    else {
        // -d IS NOT PRESENT

        // since -d is not present, only one flag to check
        if(strcmp(argv[1],"-m") == 0) {
            // ONLY -m IS PRESENT: N = argv[2]

            N = atoi(argv[2]);
            T = atoi(argv[3]);

            // CHECK N
            if(N <= 0) {
                cerr << "error: matrix dimension must be > 0 (provided size: " << N << ").\n";
                exit(2);
            }
            // CHECK T
            // range of numerical values, so must be > 0
            // can ignore whatever T is if we are processing characters
            if(T <= 0 && strcmp(argv[4],"CHAR") != 0) {
                cerr << "error: range of values must be > 0 (provided length: " << T << ").\n";
                exit(2);
            }

            // construct matrix test file
            loadMatrix(N,T,argv[4]);
        }
        else {
            // NO FLAGS ARE PRESENT: N = argv[1]

            N = atoi(argv[1]);
            T = atoi(argv[2]);

            // CHECK N
            if(N <= 0) {
                cerr << "error: amount of data must be > 0 (provided length: " << N << ").\n";
                exit(2);
            }
            // CHECK T
            // if the type is not a char, then ignore (char does not need length)
            if(strcmp(argv[3],"CHAR") != 0)
                // includes strings, so special case -1 must pass check
                if(T == 0 || T < -1) {
                    cerr << "error: range of values must not equal 0 or be less "
                         << "than -1 (provided length: " << T << ").\n";
                    exit(2);
                }

            // load test file
            loadFile(N,T,argv[3]);
        }
    }


    // terminate program
    return 0;
}



// the function takes a single argument (the shell command) and opens a pipe
// to execute said command
/*
 * pre-condition: the command must be an initialized string in main
 *
 * post-condition: the function has opened a pipe, executed a command, and
 *                 returns the output of said command, which should be an int
*/

// we are using CONST to ensure command is not altered, and passing by
// reference to avoid making a duplicate variable in memory. This approach will
// be repeated later
int executeCommand(const string& command)
{
    // DEBUG
    if(DEBUG)
        cout << "\nbeginning method: executeCommand\n";
    
    // REMOVE _ AFTER UPLOADING (_popen() -> popen())
    // open a pipe (notes) to execute command
    
    //                         (notes)
    //                            V
    FILE* pipe = popen(command.c_str(), "r");
    //        open command in read mode   ^ 

    // if pipe opening was unsuccessful, throw error
    if(!pipe) {
        //throw runtime_error("popen() failed");
        cerr << "failed to open pipe (popen() failed)).\n";
        exit(3);
    }

    int result; // variable for command result

    // read integer from command output
    //    |      specify only integer should be read
    //    |          |      store in result
    //    V          V       V
    if(fscanf(pipe, "%d", &result) != 1) {
        //                    (notes) ^
        //throw runtime_error("Failed to read integer from command output");
        cerr << "failed to read integer from command output.\n";
        exit(4);
    }


    // REMOVE _ AFTER UPLOADING (_pclose() -> pclose())
    // close the pipe, return command result
    pclose(pipe);

    // DEBUG
    if(DEBUG)
        cout << "\nend method: executeCommand\n";

    return result;
}



// this function creates a file and writes to it the matrix dimension N. If the
// matrix is not square (i.e. N != M), then M is also output. The next N lines
// are M <numerical datatype>s or characters separated by a space. 
/*
 * pre-condition: N and T must be defined and initialized to an integer value,
 *                and datatype must be initialized to a string or char*
 *
 * post-condition: nothing in main has changed. the function created an output
 *                 file and wrote to it the matrix dimensions and values
*/
void loadMatrix(const int& N, const int& T, const char* datatype)
{
    // DEBUG
    if(DEBUG)
        cout << "\nbeginning method: loadMatrix\n";
    
    // prompt for M
    int M;
    cout << "enter second matrix dimension: ";
    cin >> M;
    // CHECK M
    if(M <= 0) {
        cerr << "error: matrix dimension must be > 0 (provided size: " << M << ").\n";
        exit(2);
    }
    
    //  CONFIRMATION
    cout << "\nYou have chosen to construct:\n" << "matrix: " << N << "x" << M;
    cout << "\nmax value: " << T;
    cout << "\ntype: " << datatype;
    cout << "\n\nConfirm [Y/n]: ";
    char conf;
    cin >> conf;

    // alphabet for random string/char generation
    //char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    // promote to global?

    if(conf == 'Y') {
        // more detailed documentation found in loadFile function, the approach
        // is the same here but altered to be in matrix form

        // DEBUG
        if(DEBUG)
            cout << "\nloadMatrix: beginning file write\n";

        int testFileNum = executeCommand("ls TestFiles/matrix-test* | wc -l")+1;
        ofstream outputFile ("matrix-test"+to_string(testFileNum));
        
        // if the file cannot be created, terminate
        if(!outputFile) {
            cerr << "Failed to create output file (name used: matrix-test"
                 << testFileNum << ")." << endl;
            exit(5);
        }

        // write number of cases N to file
        // if N = M, then the matrix is square, and only one dimension will be
        // output. otherwise, output both dimensions
        if(N == M)
            outputFile << N << endl;
        else
            outputFile << N << " " << M << endl;

        
        // integer
        if(strcmp(datatype,"INT") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << rand()%T << " ";
                outputFile << endl;
            }
        // double
        else if(strcmp(datatype,"DOUBLE") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << (double)(rand()%T)/(double)(rand()%T) << " "; // REFINE THIS
                outputFile << endl;
            }
        // float
        else if(strcmp(datatype,"FLOAT") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << (float)(rand()%T)/(float)(rand()%T) << " "; // REFINE THIS
                outputFile << endl;
            }
        // char
        else if(strcmp(datatype,"CHAR") == 0)
            for(int i=0; i<N; i++) {
                for(int j=0; j<M; j++)
                    outputFile << alphabet[rand()%sizeof(alphabet)] << " ";
                outputFile << endl;
            }
        // user did not specify a valid data type (for this program)
        else {
            cerr << "error: not a valid type for this program, must be INT, "
                 << "DOUBLE, FLOAT, CHAR, or STRING" << endl;
            exit(6);
        }

        // close file
        outputFile.close();

        // move file to appropriate directory
        string moveFile = "mv matrix-test"+to_string(testFileNum)+" TestFiles";
        system(moveFile.c_str());

        // DEBUG
        if(DEBUG)
            cout << "\nloadMatrix: end file write\n";
    }
    else if(conf == 'n') {
        cout << "You have chosen to quit the program. Quitting...\n\n";
        exit(0); // not quite accurate
    }
    // this does not work how I think it will, edit later
    else {
        cout << "input not valid, respond with 'Y' or 'n': ";
        cin >> conf;
    }

    // DEBUG
    if(DEBUG)
        cout << "\nend method: loadMatrix\n";
}


// this function creates a file and first writes to it the amount of data in
// the file N. The next line consists N <numerical datatype>s or characters 
// separated by a space. Strings are generated at a random length and separated
// by a new line.
/*
 * pre-condition: N and T must be defined and initialized to an integer value,
 *                and datatype must be initialized to a string or char*
 * 
 * post-condition: nothing in main has changed. the function created an output
 *                 file and wrote to it the amount of data and each data value
*/
void loadFile(const int& N, const int& T, const char* datatype)
{
    // DEBUG
    if(DEBUG)
        cout << "\nbeginning method: loadFile\n";
    
    //  CONFIRMATION
    cout << "\nYou have chosen:\n" << "number of values: " << N;
    cout << "\nmax value: " << T;
    cout << "\n\nConfirm [Y/n]: ";
    char conf;
    cin >> conf;

    // alphabet for random string/char generation
    char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    // promote to global?
    
    // this is what the user wants to do
    if(conf == 'Y') {
        // create a file for output with corresponding test number

        // the CLI command (passed as an argument) will return the number of files
        // in "./TestFiles". This number will be used to create the next 
        // test file with the corresponding number, which is why we add 1 since we
        // do not want to create a file with a duplicate name. 

        // For example, if the command returns 2 (meaning 2 files are in the
        // TestFiles directory), then a file will be created named "test3". 
        int testFileNum = executeCommand("ls TestFiles/test* | wc -l")+1;
        // Note: system() could have been used, but I wanted to try using pipes

        // use to_string to convert commandOutput to a string, append to "test"
        // without this, the compiler thinks it's adding two numbers (strings/chars
        // can be used with integer operations in C/C++)
        ofstream outputFile ("test"+to_string(testFileNum));

        // if the file cannot be created, terminate
        if(!outputFile) {
            cerr << "Failed to create output file (name used: test"
                 << testFileNum << ")." << endl;
            exit(5);
        }

        // write number of cases N to file
        outputFile << N << endl;

        // integer
        // using strcmp to get around comparing char* and string warning (-Wall)
        //if(argv[3] == "INT")
        if(strcmp(datatype,"INT") == 0)
            for(int i=0; i<N; i++)
                outputFile << rand()%T << " "; // change " " to \n or endl if  
        // double                              // data needs to be separated 
        else if(strcmp(datatype,"DOUBLE") == 0)// by newline, may update later
            for(int i=0; i<N; i++)
                outputFile << (double)(rand()%T)/(double)(rand()%T) << " "; // REFINE THIS
        // float
        else if(strcmp(datatype,"FLOAT") == 0)
            for(int i=0; i<N; i++)
                outputFile << (float)(rand()%T)/(float)(rand()%T) << " "; // REFINE THIS
        // char
        else if(strcmp(datatype,"CHAR") == 0)
            for(int i=0; i<N; i++)
                outputFile << alphabet[rand()%sizeof(alphabet)] << " ";
        // string
        else if(strcmp(datatype,"STRING") == 0) {
            // variable string length
            if(T == -1) {
                // print N words
                for(int i=0; i<N; i++) {
                    // get random char
                    for(size_t j=0; j<rand()%sizeof(alphabet); j++)
                        outputFile << alphabet[rand()%sizeof(alphabet)];
                    // output new line instead of space, end of the word
                    outputFile << endl;
                }
            }
            // uniform string length
            else {
                // print N words
                for(int i=0; i<N; i++) {
                    // get random char
                    // using size_t to avoid warning with -Wall (notes)
                    for(int j=0; j<T; j++)
                        outputFile << alphabet[rand()%sizeof(alphabet)];
                    // output new line instead of space, end of the word
                    outputFile << endl;
                }
            }
            // VARIABLE LENGTH
            // MAJOR VARIABLE LENGTH
            // UNIFORM
        }
        // user did not specify a valid data type (for this program)
        else {
            cerr << "error: not a valid type for this program, must be INT, "
                 << "DOUBLE, or FLOAT" << endl;
            exit(6);
        }
        

        // close file
        outputFile.close();


        // use CLI to move generated file to appropriate directory
        // ORIGINAL:
        //     system("mv test"+to_string(testFileNum)+" TestFiles");
        // ChatGPT corrected this
        string moveFile = "mv test"+to_string(testFileNum)+" TestFiles";
        system(moveFile.c_str());
        // could remove this if directory included in file creation command
    }
    else if(conf == 'n') {
        cout << "You have chosen to quit the program. Quitting...\n\n";
        exit(0); // not quite accurate
    }
    // this does not work how I think it will, edit later
    else {
        cout << "input not valid, respond with 'Y' or 'n': ";
        cin >> conf;
    }

    // DEBUG
    if(DEBUG)
        cout << "\nend method: loadFile\n";
}


/* FULL LIST OF THINGS REFERENCED, RESEARCHED, OR TAKEN FROM CHATGPT
 *
 * 
 * to_string() -> taken from ChatGPT, researched at:
 * https://en.cppreference.com/w/cpp/string/basic_string/to_stringhowever
 * 
 * 
 * implementation of bash commands in a program -> taken from ChatGPT
 * 
 * 
 * general pipe understanding:
 * https://www.man7.org/linux/man-pages/man2/pipe.2.html
 * https://stackoverflow.com/questions/4812891/fork-and-pipes-in-c
 * 
 * 
 * c_str():
 * "Returns a pointer to an array that contains a null-terminated sequence of
 * characters (i.e., a C-string) representing the current value of the string
 * object." In other words, convert to a string.
 * https://cplusplus.com/reference/string/string/c_str/
 * 
 * 
 * fscanf return value:
 * "On success, the function returns the number of items of the argument list
 * successfully filled. This count can match the expected number of items or be
 * less (even zero) due to a matching failure, a reading error, or the reach of
 * the end-of-file." 
 * We want the function to return 1 item: the number of files. Therefore, the 
 * 'number of items successfully filled' should be 1. So if the function
 * returns a value other than that (1), something went wrong.
 * https://cplusplus.com/reference/cstdio/fscanf/
 * 
 * 
 * convert CLI arguments to ints (atoi, stoi) -> from ChatGPT
 * 
 * 
 * atoi -> ASCII to integer
 * stoi -> string to integer
 * https://stackoverflow.com/questions/37838417/what-do-atoi-atol-and-stoi-stand-for
 * 
 * 
 * random string generation:
 * https://stackoverflow.com/questions/47977829/generate-a-random-string-in-c11
 * 
 * 
 * size_t -> from ChatGPT
 * "The warning occurs because the type of word.length() is size_t, which is an
 * unsigned integer type, and the loop variable i is of type int, which is
 * signed"
*/
