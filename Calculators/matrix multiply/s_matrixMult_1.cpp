/* MATRIX MULTIPLICATION
 * William Wadsworth
 * CSC-4310
 * Created: at some point
 * Doctored: 5.29.2024
 * ~/csc4310/matrixmult/matrixDouble.cpp
 * 
 * This program reads matrix data stored in separate files and performs matrix multiplication. The
 * output is displayed for the user and is also output to a separate file.
 * 
 * To compile:
 * g++ s_matrixMult_1.cpp -Wall -o <exe name>
 * 
 * To run:
 * ./<exe name> <matrix1 datafile> <matrix2 datafile> <output file>
 * 
 * 
 * MATRIX FILE STRUCTURE
 * <rows> <columns>
 * nums` -->
 * |
 * V
 * 
 * `separated by a space
*/

#include <iostream>
#include <fstream>

using namespace std;


bool processCommandLine(const int&);
int** initMatrix(const string&, int&, int&);
int** mult(const int**, const int**, const int&, const int&, const int&);
void dump(const int**, const string&, const int&, const int&, const int&);


int main(int argc, char** argv)
{
    if (!processCommandLine(argc)) {
        cerr << "Error: incorrect usage of command line arguments" << endl;
        exit(1);
    }

    // --------------------------------------------------------------
    // INTRODUCTION

    cout << "This program multiplies two matrices (stored in datafiles) and outputs to the user "
         << "and a separate file specified at runtime.\n\n";

    int rows_a = INT_MIN, cols_a = INT_MIN;
    int** a = initMatrix(string(argv[1]),rows_a,cols_a);

    int rows_b = INT_MIN, cols_b = INT_MIN;
    int** b = initMatrix(string(argv[2]),rows_b,cols_b);

    // NOTE: cols_a and rows_b MUST be equal
    if(cols_a!=rows_b) {
        cerr << "error: matrix dimensions invalid for multiplication.\n";
        exit(1);
    }

    // --------------------------------------------------------------
    //  MATRIX MULTIPLICATION

    cout << "\nStarting matrix multiplication..." << endl;
    int** product = mult(a,b,rows_a,cols_a,cols_b);
    cout << "\nMatrix multiplication complete" << endl;

    // --------------------------------------------------------------
    //  OUTPUT
    
    dump(product,string(argv[3]),rows_a,rows_b,cols_b);

    return 0;
}


bool processCommandLine(const int& argc)
{
    // check if I/O redirection is used correctly
    if(argc != 4) {
        cout << "error: must have 4 arguments, " << argc << " provided.\n\n";
        cout << "Usage: ./matrixD [matrix1 datafile] [matrix2 datafile] [output file]" << endl;
        return false;
    }

    return true;
}


int** initMatrix(const string& file, int& rows, int& cols)
{
    ifstream matrixfile (file);
    
    // get dim size
    matrixfile >> rows >> cols;

    // --------------------------------------------------------------
    //  DYNAMIC ALLOCATION

    int** a = new int*[rows];
    for(int i=0; i<rows; i++)
        a[i] = new int[cols];

    // --------------------------------------------------------------
    //  READING

    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            matrixfile >> a[i][j];
        
    return a;
}


int** mult(const int** a, const int** b, const int& rows_a, const int& cols_a, const int& cols_b)
{
    // allocate
    int** product = new int*[rows_a];
    for(int i=0; i<rows_a; i++)
        product[i] = new int[cols_b];
    
    // multiply
    for(int i=0; i<rows_a; i++)
        for(int j=0; j<cols_b; j++) {
            product[i][j] = 0;

            for(int k=0; k<cols_a; k++)  
                product[i][j] += a[i][k]*b[k][j];
        }
    
    return product;
}


void dump(const int** product, const string& file, const int& rows_a, const int& rows_b, const int& cols_b)
{
    ofstream ans (file);

    ans << rows_a << " " << cols_b << "\n";
    for(int i=0; i<rows_a; i++) {
        for(int j=0; j<cols_b; j++) {
            ans << product[i][j] << " ";
            cout << product[i][j] << " ";
        }
        ans << endl;
        cout << endl;
    }
    ans.close();

    // free memory
    for(int i=0; i<rows_a; i++)
        delete[] product[i];
    delete[] product;
}

// NOTES -- all from StackOverflow

/* You want to use endl to end your lines. An alternative is using '\n' character. 
 * These two things are different, endl flushes the buffer and writes your output 
 * immediately while '\n' allows the outfile to put all of your output into a buffer
 * and maybe write it later.
 * 
 * "flushes output buffer"?
*/

/* Consider writing to a file. This is an expensive operation. If in your code you write
 * one byte at a time, then each write of a byte is going to be very costly. So a common
 * way to improve performance is to store the data that you are writing in a temporary 
 * buffer. Only when there is a lot of data is the buffer written to the file. By
 * postponing the writes, and writing a large block in one go, performance is improved.
 * With this in mind, flushing the buffer is the act of transferring the data from the
 * buffer to the file. It clears the buffer by outputting everything in it.
 * 
 * CIN FLUSHES COUT - Reading cin happens when you use the stream operator to read from
 * cin. Typically you want to flush cout when you read because otherwise the input may
 * appear before the prompt.
*/

/* A flattened array means you use just a single dimentional array that has the same
 * number of elements as your desired 2D array, and you perform arithmetic for converting
 * between the multi-dimensional indices and the corresponding single dimensional index.
 *
 * Row i, column j in the 2d array corresponds to { arr[column_count*i + j] }. { arr[n] }
 * corresponds to the element at row { n/column_count } and column { n% column_count }.
 * For example, in an array with 10 columns, row 0 column 0 corresponds to { arr[0] }.
 * row 0, column 1 correponds to { arr[1] }. row 1 column 0 correponds to { arr[10] }.
 * row 1, column 1 corresponds to { arr[11] }.
 * 
 * NOTE - I think this approach can work, but I will save it for another day.
*/

/* INTERESTING NOTE - went from using 1D product array to 2D and forgot to change output
 * loops. Instead of numbers, it was outputting memory addresses.
 *
 * I had this:              cout << product[j] << " ";
 * when it should be:       cout << product[i][j] << " ";
*/