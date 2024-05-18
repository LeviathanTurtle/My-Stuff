/* SUM MATRIX QUADRANTS
 * William Wadsworth
 * CSC 4310
 * 10.31.2023
 * 
 * [DESCRIPTION]:
 * This program uses CUDA to do an element-wise sum of a matrix, and stores the
 * answers in the second quadrant.
 *
 * [COMPILE/RUN]:
 * To compile:
 *     nvcc sumQuads.cu -o sq
 * To run:
 *     ./sq <matrix file> <output file>
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - the user confirmed information is incorrect or the program completed one
 *     full execution
 *
 * 1 - CL arguments were used incorrectly
 *
 * 2 - file was unable to be opened or created
*/

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

// function prototypes
//                filename    size
int* loadMatrix(const char*, int&);
//                     matrix   sum   size
__global__ void sumKernel(int*, int*, int);
//     matrix  size
void sum(int*, int);
//         filename   matrix  size
void dump(const char*, int*, int&);


int main(int argc, char* argv[])
{
    // check CL arguments
    if(argc != 3) {
        cerr << "error: invalid arguments. must be: exe matrixFile outputFile.\n";
        exit(1);
    }

    // variable for matrix size
    int size;

    // load matrix, pass filename
    int* matrix = loadMatrix(argv[1],size);

    sum(matrix,size);

    dump(argv[2],matrix,size);

    // deallocate matrix
    delete[] matrix;

    return 0;
}


// This function creates, loads, and returns a 1-D array from a file
/* pre-condition: filename must be set up from argv and size must be a declared
 *                integer
 *
 * post-condition: the newly constructed matrix is returned and the input file
 *                 is closed
*/
int* loadMatrix(const char* filename, int& size)
{
    // create file
    ifstream inputFile (filename);

    // check file was opened
    if(!inputFile) {
        cerr << "error: file unable to be opened or created.\n";
        exit(2);
    }

    // read in matrix size
    inputFile >> size;

    // create matrix
    int* matrix = new int [size*size];

    // read in data
    for(int i=0; i<size*size; i++)
        inputFile >> matrix[i];
    inputFile.close();

    return matrix;
}


// This function is the CUDA kernel function that performs the element-wise sum
// on the passed matrix. This is run by each thread.
/* pre-condition: matrix, ans, and n must be initialized
 *
 * post-condition: the second quadrant of the matrix is updated with the sum
*/
__global__ void sumKernel(int* matrix, int* ans, int n)
{
    // setup thread IDs
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    // ensure we are 
    if(row < n/2 && col < n/2) {
        // avoid RC by ensuring addition is atomic
        atomicAdd(ans, matrix[row*n+col]);
    }
}


// This function sets up the matrix and answer value to be used by the CUDA
// kernel.
/* pre-condition: A (matrix) and its size n must be initialized
 *
 * post-condition: the calculated sums are copied into the matrix A and the
 *                 temporary matrices and values are deallocated
*/
void sum(int* A, int n)
{
    // variable for n/2 so we perform calc once
    int halfsize = n/2;
    // calculate new size
    int size = halfsize * halfsize * sizeof(int);

    // answer matrix
    int* A_d;
    // allocate memory for answer matrix
    cudaMalloc((void**) &A_d, n*n*sizeof(int));
    // copy contents of A into answer matrix
    cudaMemcpy(A_d, A, n*n*sizeof(int), cudaMemcpyHostToDevice);

    // variable for answer
    int* ans_d;
    // allocate memory for answer
    cudaMalloc((void**) &ans_d, size);

    // block and grid size
    dim3 blockSize(16,16);
    dim3 gridSize(32, 32);

    sumKernel<<<gridSize, blockSize>>>(A_d, ans_d, n);

    // copy contents of answer array into A
    cudaMemcpy(A, ans_d, size, cudaMemcpyDeviceToHost);

    // deallocate memory
    cudaFree(A_d);
    cudaFree(ans_d);
}


// This function outputs the contents of the matrix to a file
/* pre-condition: filename must be set up from argv and size must be a declared
 *                integer
 *
 * post-condition: the matrix size N/2 is output and the next N/2 lines are 
 *                 filled with N/2 integers (the element-wise sums)
*/
void dump(const char* filename, int* matrix, int& size)
{
    // create output file with filename passed
    ofstream outputFile (filename);

    // if output file was not able to be opened, error and exit
    if(!outputFile) {
        cerr << "error: file unable to be created opened\n";
        exit(2);
    }

    // write size to file to maintain consistent file structure
    outputFile << size/2 << endl;
    // output the contents of the array
    for(int i=0; i<size/2; i++) {
        for(int j=0; j<size/2; j++)
            outputFile << matrix[i*size+j] << " ";
        outputFile << endl;
    }
    
    outputFile.close();
}
