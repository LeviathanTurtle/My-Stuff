/* MATRIX MULTIPLICATION USING OPENMP
 * William Wadsworth
 * CSC-4310
 * Created: 10.3.23
 * Doctored: 5.29.2024
 * ~/csc4310/midterm/matrixCplus-Bminus.cpp
 * 
 * This program reads matrix data stored in separate files and performs matrix multiplication. The
 * calculation is done using OpenMP. The output is displayed for the user and is also output to a
 * separate file.
 * 
 * To compile:
 * g++ p_matrixMult_openmp.cpp -Wall -fopenmp -o p_matrixMult_openmp
 * 
 * To run:
 * ./p_matrixMult_openmp [matrix1 datafile] [matrix2 datafile] [output file] [num threads]
 * 
 * 
 * MATRIX FILE STRUCTURE
 * [size]
 * nums* -->
 * |
 * V
 * 
 * *separated by a space. Note that this program assumes matrices are square
 * 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - incorrect usage of program arguments
 * 
 * 2 - matrix data/output files unable to be opened
 * 
 * 3 - invalid matrix dimensions (incompatible)
*/

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <omp.h>

using namespace std;

// global bool for debug options. This way I won't have to pass in between
// functions. This is for my sanity
bool DEBUG = false;


int** allocateMatrix(const int&);
void deallocateMatrix(const int**, const int&);
void loadMatrix(ifstream&, int**, const int&);
void mult(const int**, const int**, int**, const int&, const int&);
void printMatrix(const string&, const int**, const int&);


int main(int argc, char** argv)
{
    // check if I/O redirection is used correctly (detailed in opening comment)
    if(argc != 5) {
        cerr << "error: must have 5 arguments, " << argc << " provided.\nUsage: "
             << "./matrixCplus-Bminus [matrix1 datafile] [matrix2 datafile] [output file] "
             << "[num threads]" << endl;
        exit(1);
    }

    // --------------------------------------------------------------
    //  INTRODUCTION

    cout << "This program multiplies two matrices (stored in datafiles) and "
         << "outputs to the user and a separate file specified at runtime.\n\n";

    // --------------------------------------------------------------
    // OPENING FILES

    string file1 = argv[1], file2 = argv[2];

    // open files (names passed from main)
    if(DEBUG)
        cout << "\nopening files..." << endl;
    ifstream matrixFile1 (file1), matrixFile2 (file2);

    // check if files were opened successfully
    if(!matrixFile1) {
        cerr << "error: first matrix file '" << file1 << "' unable to be opened\n";
        exit(2);
    }
    if(!matrixFile2) {
        cerr << "error: second matrix file '" << file2 << "' unable to be opened\n";
        exit(2);
    }

    if(DEBUG)
        cout << "files successfully opened." << endl;

    // --------------------------------------------------------------
    // READING DIMENSIONS

    if(DEBUG)
        cout << "\nreading dimensions..." << endl;

    // get dim sizes from respective files
    int dim1 = INT_MIN, dim2 = INT_MIN;
    matrixFile1 >> dim1;
    matrixFile2 >> dim2;

    // NOTE: dim1 and dim2 MUST be equal or matrix mult is invalid
    if(dim1!=dim2) {
        cerr << "error: matrix dimensions invalid for multiplication.\n";
        exit(3);
    }

    if(DEBUG)
        cout << "dimensions successfully read." << endl;

    // unified dimension variable because IT SHOULD BE SQUARE, DIMS SHOULD BE
    // THE SAME
    int dim = dim1;

    // --------------------------------------------------------------
    //  DYNAMIC ALLOCATION

    if(DEBUG)
        cout << "\ndynamically creating arrays A, B, C..." << endl;

    // first matrix
    int** a = allocateMatrix(dim);
    // second matrix
    int** b = allocateMatrix(dim);
    // product array
    int** product = allocateMatrix(dim);
    // note that since the sizes should be the same (multiplying two square
    // matricies), the sizes defined are interchangeable (i.e. dim1 = dim2)

    if(DEBUG)
        cout << "arrays successfully created." << endl;

    // --------------------------------------------------------------
    //  READING FROM FILES

    if(DEBUG)
        cout << "\nreading data from matrix files..." << endl;
    
    // fill matrices with data
    loadMatrix(matrixFile1, a, dim);
    loadMatrix(matrixFile2, b, dim);
    // note: not ensuring product is array of 0's because that is handled in mult function

    // close files - they are no longer needed
    matrixFile1.close();
    matrixFile2.close();

    if(DEBUG)
        cout << "data successfully read." << endl;

    // --------------------------------------------------------------
    //  MATRIX MULTIPLICATION

    // stuff for finding time difference
    struct timeval startTime, stopTime;
    double start, stop;

    if(DEBUG)
        cout << "\nbeginning matrix multiplication..." << endl;
    
    // matrix mult
    gettimeofday(&startTime,NULL);
    mult(a,b,product,dim,atoi(argv[4]));
    gettimeofday(&stopTime,NULL);

    // find time difference
    start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
    stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0); 

    if(DEBUG)
        cout << "matrix multiplication successfully completed." << endl;

    // --------------------------------------------------------------
    //  OUTPUT

    string outFile = argv[3];
    printMatrix(outFile,product,dim);
    cout << "The result was found in " << stop-start << "s" << endl;

    // --------------------------------------------------------------
    //  TERMINATION

    // free memory
    deallocateMatrix(a,dim);
    deallocateMatrix(b,dim);
    deallocateMatrix(product,dim);
    
    return 0;
}


int** allocateMatrix(const int& n) 
{
    int** matrix = new int*[n];
    for(int i=0; i<n; i++)
        matrix[i] = new int[n];
    
    return matrix;
}


void deallocateMatrix(const int** matrix, const int& n)
{
    for(int i=0; i<n; i++)
        delete[] matrix[i];
    delete[] matrix;
}


void loadMatrix(ifstream& inputFile, int** matrix, const int& n)
{
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            inputFile >> matrix[i][j];
}


void mult(const int** a, const int** b, int** product, const int& dim, const int& T)
{
    // run this block in T threads
    #pragma omp parallel num_threads(T)
    {
        // each thread gets their ID
        int myrank = omp_get_thread_num();
        // each thread gets their amount of data
        int size = dim/T;
        // get the first index for current thread. myrank is unique to each
        // thread, so startRow is also unique to each thread 
        int startRow = myrank * size;
        // same thing, but end of row index
        if(myrank == T-1)     // if current thread is the last one, we know the
            int endRow = dim; // end index is dim
        else
            int endRow = (myrank+1)*size; // otherwise, find the end index
        //int endRow = (myrank == T-1) ? dim : (myrank+1)*size;

        // row
        for(int i=startRow; i<endRow; i++)
            // column
            for(int j=0; j<dim; j++) {
                // ensure adding to 0 and nothing wonky
                product[i][j] = 0;

                // update cell with correct product
                // omp critical to avoid race condition
                #pragma omp critical
                for(int k=0; k<dim; k++)  
                    product[i][j] += a[i][k]*b[k][j];
            }
    }        
}


void printMatrix(const string& outputFile, const int** array, const int& dim)
{
    // --------------------------------------------------------------
    //  CREATING OUTPUT FILE

    if(DEBUG)
        cout << "\nopening output file..." << endl;

    // create answer file
    ofstream ans (outputFile);

    // check if file was opened successfully
    if(!ans) {
        cerr << "matrix file '" << outputFile << "' unable to be opened\n";
        exit(2);
    }

    if(DEBUG)
        cout << "output file successfully opened." << endl;

    // --------------------------------------------------------------
    //  OUTPUT FOR USER AND/OR FILE

    if(DEBUG)
        cout << "\noutputting..." << endl;

    // does the user want to see the output as well?
    cout << "The product will be output to a file similar to the input. Would "
         << "you also like the output to be shown here? [Y/n]: ";
    char confirmation;
    cin >> confirmation;

    ans << dim << endl;
    cout << endl << dim << endl;

    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++) {
            ans << array[i][j] << " ";
            if(confirmation == 'Y')
                cout << array[i][j] << " ";
        }
        ans << endl;
        if(confirmation == 'Y')
            cout << endl;
    }

    if(DEBUG)
        cout << "output successful." << endl;

    ans.close();
}

