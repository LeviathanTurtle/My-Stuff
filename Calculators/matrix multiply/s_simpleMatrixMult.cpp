/* MATRIX MULTIPLICATION USING 1-D ARRAYS
 * William Wadsworth
 * CSC 4310
 * Created: at some point
 * Doctored: 5.27.2024
 * 
 * This simple program creates two matrices filled with random numbers between 0-9 and writes them
 * to separate files. It then performs matrix multiplication on them and writes the product matrix
 * to a new file, as well as displays it for the user.
 * 
 * To compile:
 * g++ simpleMatrixMult.cpp -Wall -o simpleMatrixMult
 * 
 * To run:
 * ./simpleMatrixMult
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


void multInit();
char getDimensions(string&, string&, int&, int&, int&);
void mult(const string&, const string&);


int main()
{
    // INTRODUCTION
    cout << "This function creates two matrices of user-defined size for multiplcation.\n\n";
    
    multInit();

    return 0;
}


void multInit()
{
    int dim1, dim2, dim3;
    string matrixfile1, matrixfile2;

    if(getDimensions(matrixfile1,matrixfile2,dim1,dim2,dim3) == 'Y') {
        cout << "Processing...\n";

        // MATRIX CONSTRUCTION
        ofstream outfile1 (matrixfile1);
        ofstream outfile2 (matrixfile2);

        srand(time(NULL)); // seed RNG

        // write to respective files
        outfile1 << dim1 << " " << dim2 << endl;
        for(int i=0; i<dim1; i++) {
            for(int j=0; j<dim2; j++)
                outfile1 << rand()%10 << " "; // value range: 0-9
            outfile1 << endl;
        }
        outfile1.close();

        outfile2 << dim2 << " " << dim3 << endl;
        for(int i=0; i<dim2; i++) {
            for(int j=0; j<dim3; j++)
                outfile2 << rand()%10 << " "; // value range: 0-9
            outfile2 << endl;
        }

        outfile2.close();
        cout << "Done.\n";
    }
    else
        exit(1);


    // MATRIX MULTIPLICATION
    mult(matrixfile1,matrixfile2);
}


char getDimensions(string& matrixfile1, string& matrixfile2, int& dim1, int& dim2, int& dim3)
{
    // USER NAMES FILES
    cout << "Name first matrix file: ";
    cin >> matrixfile1;
    cout << "Name second matrix file: ";
    cin >> matrixfile2;

    // MATRIX DIMENSIONS
    cout << "matrix1 row dim: ";
    cin >> dim1;
    cout << "matrix1 col dim: ";
    cin >> dim2;
    cout << "matrix2 col dim: ";
    cin >> dim3;

    // CONFIRMATION
    cout << "\nYou have chosen:\n" << "matrix1: " << dim1 << "x" << dim2;
    cout << "\nmatrix2: " << dim2 << "x" << dim3;
    cout << "\nConfirm [Y/n]: ";
    char conf;
    cin >> conf;

    return conf;
}


void mult(const string& file1, const string& file2)
{
    // open files (names passed from main)
    ifstream m1 (file1), m2 (file2);

    // get dim sizes from respective files
    int row1, col1, row2, col2;
    m1 >> row1 >> col1; // 3 2
    m2 >> row2 >> col2; // 2 2
    // NOTE: col1 and row2 MUST be equal

    // dynamically allocate array size
    int* a = new int[row1*col1];
    int* b = new int[row2*col2];
    // read in data from first file
    for(int i=0; i<row1*col1; i++) {
        m1 >> a[i];
        //cout << a[i] << " ";
    }
    //cout << endl;
    m1.close();

    // repeat with second file
    for(int i=0; i<row2*col2; i++) {
        m2 >> b[i];
        //cout << b[i] << " ";
    }
    //cout << endl;
    m2.close();
        
    // define product array
    int prod_size = row1*col2;
    int* product = new int[prod_size]; // int[6]

    // calculate
    for(int i=0; i<row1; i++)
        for(int j=0; j<col2; j++) {
            product[col2*i+j] = 0;

            for(int k=0; k<row2; k++)  
                product[row1*i+j] += a[row1*i+k] * b[row2*k+j];
        }
    // free memory
    delete[] a;
    delete[] b;

    // output final product
    ofstream ans ("answer");
    ans << row1 << " " << col2 << endl;
    for(int i=0; i<row1; i++) {
        for(int j=0; j<col2; j++) {
            int index = i * col2 + j;
            ans << product[index] << " ";
            cout << product[index] << " ";
        }
        ans << endl;
        cout << endl;
    }
    
    // free memory
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
 * buffer. Only when there is a lot of data is the buffer written to the file. By postponing 
 * the writes, and writing a large block in one go, performance is improved. With this in
 * mind, flushing the buffer is the act of transferring the data from the buffer to the file.
 * It clears the buffer by outputting everything in it.
 * 
 * CIN FLUSHES COUT - Reading cin happens when you use the stream operator to read from cin. 
 * Typically you want to flush cout when you read because otherwise the input may appear before 
 * the prompt.
*/

/* A flattened array means you use just a single dimentional array that has the same number of 
 * elements as your desired 2D array, and you perform arithmetic for converting between the 
 * multi-dimensional indices and the corresponding single dimensional index.
 *
 * Row i, column j in the 2d array corresponds to { arr[column_count*i + j] }. { arr[n] } corresponds 
 * to the element at row { n/column_count } and column { n% column_count }. For example, in an array with 
 * 10 columns, row 0 column 0 corresponds to { arr[0] }. row 0, column 1 correponds to { arr[1] }. row 1 
 * column 0 correponds to { arr[10] }. row 1, column 1 corresponds to { arr[11] }.
 */