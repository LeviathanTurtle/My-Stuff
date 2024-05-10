/* MATRIX MULTIPLICATION
 * William Wadsworth
 * CSC-4310
 * ~/csc4310/matrixmult/matrixDouble.cpp
 * 
 * This program reads matrix data stored in separate files and performs matrix
 * multiplication. The output is displayed for the user and is also output to a separate
 * file.
 * 
 * To compile:
 * g++ matrixDouble.cpp -Wall -o matrixD
 * 
 * To run:
 * ./matrixD [matrix1 datafile] [matrix2 datafile] [output file]
 * 
 * 
 * MATRIX FILE STRUCTURE
 * [rows] [columns]
 * nums* -->
 * |
 * V
 * 
 * *separated by a space
*/

#include <iostream>
#include <fstream>

using namespace std;

void mult(int** a, int** b, int** product, int rowsa, int colsa, int colsb)
{
    for(int i=0; i<rowsa; i++)
        for(int j=0; j<colsb; j++) {
            product[i][j] = 0;

            for(int k=0; k<colsa; k++)  
                product[i][j] += a[i][k]*b[k][j];
        }        
}

int main(int argc, char** argv)
{
    // check if I/O redirection is used correctly (detailed in opening comment)
    if(argc != 4) {
        cout << "error: must have 4 arguments, " << argc << " provided. check opening comment.\n" << endl;
        return 1;
    }
    
    // --------------------------------------------------------------
    //  INTRODUCTION

    cout << "This program multiplies two matrices (stored in datafiles) and outputs to the user and a separate file specified at runtime.\n\n";

    // --------------------------------------------------------------
    //  READING FILES

    ifstream m1, m2;

    // open files (names passed from main)
    m1.open(argv[1]); // not sure if this is 'valid' or not. Specifically, you can call
    m2.open(argv[2]); // main, but should you?

    // get dim sizes from respective files
    int row1, col1, row2, col2;
    m1 >> row1 >> col1;
    m2 >> row2 >> col2;
    // NOTE: col1 and row2 MUST be equal
    if(col1!=row2) {
        cout << "error: matrix dimensions invalid for multiplication.\n";
        return 1;
    }

    // --------------------------------------------------------------
    //  DYNAMIC ALLOCATION

    // first matrix
    int** a = new int*[row1];
    for(int i=0; i<row1; i++)
        a[i] = new int[col1];
    // second matrix
    int** b = new int*[row2];
    for(int i=0; i<row2; i++)
        b[i] = new int[col2];
    // product array
    int** product = new int*[row1];
    for(int i=0; i<row1; i++)
        product[i] = new int[col2];

    // --------------------------------------------------------------
    //  READING

    // first matrix
    for(int i=0; i<row1; i++)
        for(int j=0; j<col1; j++)
            m1 >> a[i][j];
    // second matrix
    for(int i=0; i<row2; i++)
        for(int j=0; j<col2; j++)
            m2 >> b[i][j]; 

    // --------------------------------------------------------------
    //  MATRIX MULTIPLICATION

    cout << "\nStarting matrix multiplication..." << endl;
    mult(a,b,product,row1,col1,col2);
    cout << "\nMatrix multiplication complete" << endl;

    // --------------------------------------------------------------
    //  OUTPUT
    ofstream ans;
    ans.open(argv[3]);
    ans << row1 << " " << col2 << endl;
    for(int i=0; i<row1; i++) {
        for(int j=0; j<col2; j++) {
            ans << product[i][j] << " ";
            cout << product[i][j] << " ";
        }
        ans << endl;
        cout << endl;
    }

    // free memory, close program
    for(int i=0; i<row1; i++) {
        delete[] product[i];
        delete[] a[i];
    }
    for(int i=0; i<row2; i++)
        delete[] b[i];
        
    m1.close();
    m2.close();
    return 0;
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