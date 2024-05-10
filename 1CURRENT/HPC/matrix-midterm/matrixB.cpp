/* MATRIX MULTIPLICATION USING MPI
 * William Wadsworth
 * CSC-4310
 * 10.3.23
 * ~/csc4310/midterm/matrixB.cpp
 * 
 * 
 * This program reads matrix data stored in separate files and performs matrix
 * multiplication. The calculation is done using MPI. The output is displayed
 * for the user and is also output to a separate file.
 * 
 * The master process uses a modified version of the Strassen algorithm to
 * recursively carve up the matrix data at the top level into 8 multiplications,
 * then uses OpenMPI to distribute to 8 worker processes. Each worker performs
 * the multiplications of the data it receives from the master. The worker
 * process then sends its solution back to the master. The master process 
 * calculates the final product array and is output to a file and/or to the 
 * screen.
 * 
 * 
 * To compile:
 * mpicc matrixB.cpp -Wall -o matrixB
 * 
 * To run:
 * mpiexec --mca btl_tcp_if_include <ethernet_ID> -n 4 -hostfile actHosts
 *         [matrix1 datafile] [matrix2 datafile] [output file]
 * 
 * 
 * MATRIX FILE STRUCTURE
 * [size]
 * nums* -->
 * |
 * V
 * 
 * *separated by a space. This program assumes both matrices are square
*/

#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <mpi.h>

using namespace std;


// global bool for debug options. This way I won't have to pass in between
// functions. This is for my sanity
bool DEBUG = false;


int** allocateMatrix(int n) 
{
    int** matrix = new int*[n];
    for(int i=0; i<n; i++)
        matrix[i] = new int[n];
    
    return matrix;
}


void deallocateMatrix(int** matrix, int n)
{
    for(int i=0; i<n; i++)
        delete[] matrix[i];
    delete[] matrix;
}


void loadMatrix(ifstream& inputFile, int** matrix, int n)
{
    for(int i=0; i<n; i++)
        for(int j=0; j<n; j++)
            inputFile >> matrix[i][j];
}


void mult(int** A, int** B, int** C, int dim)
{
    if(dim == 1)
        C[0][0] = A[0][0] * B[0][0];
    else {
        // create temp matrix T
        int** T = allocateMatrix(dim);

        // submatrix arrays
        int newSize = dim/2;
        // this is ugly I hate this
        int** subA1_1 = allocateMatrix(newSize);
        int** subA1_2 = allocateMatrix(newSize);
        int** subA2_1 = allocateMatrix(newSize);
        int** subA2_2 = allocateMatrix(newSize);

        int** subB1_1 = allocateMatrix(newSize);
        int** subB1_2 = allocateMatrix(newSize);
        int** subB2_1 = allocateMatrix(newSize);
        int** subB2_2 = allocateMatrix(newSize);

        int** subC1_1 = allocateMatrix(newSize);
        int** subC1_2 = allocateMatrix(newSize);
        int** subC2_1 = allocateMatrix(newSize);
        int** subC2_2 = allocateMatrix(newSize);

        int** subT1_1 = allocateMatrix(newSize);
        int** subT1_2 = allocateMatrix(newSize);
        int** subT2_1 = allocateMatrix(newSize);
        int** subT2_2 = allocateMatrix(newSize);
        // I can't wait for you to tell me there's an easier solution/way to do
        // this

        // partitioning algorithm -- from ChatGPT
        // modified to update array instead of output
        for(int i=0; i<newSize; i++)
            for(int j=0; j<newSize; j++) {
                //cout << "dim/2 at (" << i << ", " << j << "):" << endl;
                subA1_1[i][j] = A[i][j];
                subA1_2[i][j] = A[i][j+newSize];
                subA2_1[i][j] = A[i+newSize][j];
                subA2_2[i][j] = A[i+newSize][j+newSize];

                subB1_1[i][j] = B[i][j];
                subB1_2[i][j] = B[i][j+newSize];
                subB2_1[i][j] = B[i+newSize][j];
                subB2_2[i][j] = B[i+newSize][j+newSize];

                subC1_1[i][j] = C[i][j];
                subC1_2[i][j] = C[i][j+newSize];
                subC2_1[i][j] = C[i+newSize][j];
                subC2_2[i][j] = C[i+newSize][j+newSize];

                subT1_1[i][j] = T[i][j];
                subT1_2[i][j] = T[i][j+newSize];
                subT2_1[i][j] = T[i+newSize][j];
                subT2_2[i][j] = T[i+newSize][j+newSize];
            }

        
        // if one the the parameters on this is wrong I'm gonna cry I've looked
        // at this for at least 15 minutes to make sure it's correct
        mult(subC1_1,subA1_1,subB1_1,newSize);
        mult(subC1_2,subA1_1,subB1_2,newSize);
        mult(subC2_1,subA2_1,subB1_1,newSize);
        mult(subC2_2,subA2_1,subB1_2,newSize);
        
        mult(subT1_1,subA1_2,subB2_1,newSize);
        mult(subT1_2,subA1_2,subB2_2,newSize);
        mult(subT2_1,subA2_2,subB2_1,newSize);
        mult(subT2_2,subA2_2,subB2_2,newSize);


        // add T to C
        for(int i=1; i<dim; i++)
            for(int j=1; j<dim; j++)
                C[i][j] += T[i][j];
    }
}


void printMatrix(string outputFile, int** array, int dim)
{
    // --------------------------------------------------------------
    //  CREATING OUTPUT FILE

    if(DEBUG)
        cout << "\nopening output file..." << endl;

    // create answer file
    ofstream ans;
    ans.open(outputFile);

    // check if file was opened successfully
    if(!ans) {
        cerr << "matrix file unabled to be opened. provided name: " << outputFile << "\n";
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

    // close file, no longer needed
    ans.close();
}


int main(int argc, char** argv)
{
    // --------------------------------------------------------------
    // MPI INITIALIZATION

    if(DEBUG)
        cout << "\nInitializing OpenMPI..." << endl;

    MPI_Init(&argc,&argv);

    // rank ID and number of processes
    int taskid, numtasks;
    // initialize number of processes
    MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
    if (numtasks % 4 != 0) {
        cout << "Quitting. Number of MPI tasks must be divisible by 4.\n";
        // quit MPI
        MPI_Abort(MPI_COMM_WORLD,1);
        exit(1);
    }
    // initialize task IDs
    MPI_Comm_rank(MPI_COMM_WORLD,&taskid);

    if(DEBUG)
        cout << "OpenMPI successfully initialized." << endl;

    

    // --------------------------------------------------------------
    // ONLY FOR MASTER PROCESS
    // this is not modular, I was lazy

    // unified dimension variable, defined here so it can be accessed by all 
    // processes
    int dim;

    if(taskid == 0) {
        // check if I/O redirection is used correctly (detailed in opening comment)
        if(argc != 11) {
            cerr << "error: must have 11 arguments, " << argc << " provided. check "
                 << "opening comment.\n" << endl;
            // quit MPI
            MPI_Abort(MPI_COMM_WORLD,1);
            exit(1);
        }

        // --------------------------------------------------------------
        //  INTRODUCTION

        cout << "This program multiplies two matrices (stored in datafiles) and "
             << "outputs to the user and a separate file specified at runtime.\n\n";
        

        // --------------------------------------------------------------
        // OPENING FILES

        ifstream matrixFile1, matrixFile2;
        string file1 = argv[1], file2 = argv[2];

        if(DEBUG)
            cout << "\nopening files..." << endl;

        // open files (names passed from main)
        matrixFile1.open(file1);
        matrixFile2.open(file2);

        // check if files were opened successfully
        if(!matrixFile1) {
            cerr << "first matrix file unabled to be opened. provided name: " << file1 << "\n";
            // quit MPI
            MPI_Abort(MPI_COMM_WORLD,2);
            exit(2);
        }
        if(!matrixFile2) {
            cerr << "second matrix file unabled to be opened. provided name: " << file2 << "\n";
            // quit MPI
            MPI_Abort(MPI_COMM_WORLD,2);
            exit(2);
        }

        if(DEBUG)
            cout << "files successfully opened." << endl;
        

        // --------------------------------------------------------------
        // READING DIMENSIONS

        if(DEBUG)
            cout << "\nreading dimensions..." << endl;

        // get dim sizes from respective files
        int dim1, dim2;
        matrixFile1 >> dim1;
        matrixFile2 >> dim2;

        // NOTE: dim1 and dim2 MUST be equal or matrix mult is invalid
        if(dim1!=dim2) {
            cerr << "error: matrix dimensions invalid for multiplication.\n";
            // quit MPI
            MPI_Abort(MPI_COMM_WORLD,3);
            exit(3);
        }
        // can't multiply a matrix that does not exist
        else if(dim1 == 0 || dim2 == 0) {
            cerr << "error: matrix dimensions must be greater than 0.\n";
            // quit MPI
            MPI_Abort(MPI_COMM_WORLD,3);
            exit(3);
        }

        if(DEBUG)
            cout << "dimensions successfully read." << endl;

        // unified dimension variable because IT SHOULD BE SQUARE, DIMS SHOULD BE
        // THE SAME
        dim = dim1;

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
        // ensure product array is 0 and nothing wonky
        for(int i=0; i<dim; i++)
            for(int j=0; j<dim; j++)
                product[i][j] = 0;

        if(DEBUG)
            cout << "arrays successfully created." << endl;

        // --------------------------------------------------------------
        //  READING FROM FILES

        if(DEBUG)
            cout << "\nreading data from matrix files..." << endl;

        
        // fill matrices with data
        loadMatrix(matrixFile1, a, dim);
        loadMatrix(matrixFile2, b, dim);

        if(DEBUG)
            cout << "data successfully read." << endl;

        // close files - they are no longer needed
        matrixFile1.close();
        matrixFile2.close();
    }



    // --------------------------------------------------------------
    //  THE REST IS FOR ALL PROCESSES

    // calculate buffer size for amount of data to send, make new receive
    // arrays
    int buffer = dim/numtasks;
    int** receiveA = allocateMatrix(buffer);
    int** receiveB = allocateMatrix(buffer);
    int** receiveC = allocateMatrix(buffer);

    // (send buffer, send count, send type, receive buffer, receive count, receive type, master, comm)
    // send chunks of data to worker processes, 0 is master process
    MPI_Scatter(a,buffer,MPI_INT,receiveA,buffer,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Scatter(b,buffer,MPI_INT,receiveB,buffer,MPI_INT,0,MPI_COMM_WORLD);

    // --------------------------------------------------------------
    //  MATRIX MULTIPLICATION

    // stuff for finding time difference
    struct timeval startTime, stopTime;
    double start, stop;

    if(DEBUG)
        cout << "\nbeginning matrix multiplication..." << endl;
    
    // matrix mult
    gettimeofday(&startTime,NULL);
    mult(a,b,product,dim);
    gettimeofday(&stopTime,NULL);

    // find time difference
    start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
    stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0); 

    if(DEBUG)
        cout << "matrix multiplication successfully completed." << endl;


    // (send buffer, send count, send type, receive buffer, receive count, receive type, master, comm)
    MPI_Gather(receiveC,buffer,MPI_INT,product,buffer,MPI_INT,0,MPI_COMM_WORLD);


    // --------------------------------------------------------------
    //  OUTPUT
    
    // we only want master process to run this portion
    if(taskid == 0) {
        string outFile = argv[3];
        printMatrix(outFile,product,dim);
        cout << "The result was found in " << stop-start << endl;
    }



    // --------------------------------------------------------------
    //  TERMINATION

    // free memory
    deallocateMatrix(a,dim);
    deallocateMatrix(b,dim);
    deallocateMatrix(product,dim);

    deallocateMatrix(receiveA,buffer);
    deallocateMatrix(receiveB,buffer);
    deallocateMatrix(receiveC,buffer);

    // bring all threads back together
    MPI_Finalize();
    
    // close program
    return 0;
}



// NOTES -- 

