/*
 *
 *
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
__global__ void sumKernel(int*, int*, int);
void sum(int*, int);
void dump(const char*, int*, int&);


int main(int argc, char* argv[])
{
    // check CL arguments
    if(argc != 3) {
        cerr << "error: invalid arguments. must be: exe inputFile outputFile.\n";
        exit(1);
    }

    //variable for matrix size
    int size;

    // load matrix, pass filename
    int* matrix = loadMatrix(argv[1],size);

    sum(matrix,size);

    dump(argv[2],matrix,size);

    // deallocate matrix
    delete[] matrix;

    return 0;
}


int* loadMatrix(const char* filename, int& size)
{
    // create file
    ifstream inputFile (filename);

    // check file was opened
    if(!inputFile) {
        cerr << "error: file unable to be opened or created.\n";
        exit(2);
    }

    //if (!(input >> matrix[i][j]))

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


// matrix is stored in linear memory (1D), ans is a single value initialized to
// 0 in caller to sum
__global__ void sumKernel(int* matrix, int* ans, int n)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if(row < n/2 && col < n/2)
        //atomicAdd(&ans,matrix[row*n+col]);
        
        //atomicAdd(ans,matrix[row*n+col]);

        atomicAdd(&ans[row*n+col], matrix[row*n+col]);
}


void sum(int* A, int n)
{
    int halfsize = n/2;
    int size = halfsize * halfsize * sizeof(int);

    int* A_d;
    cudaMalloc((void**) &A_d, size);
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

    int ans = 0;
    int* ans_d;
    cudaMalloc((void**) &ans_d, sizeof(int));
    cudaMemcpy(ans_d, &ans, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockSize(16,16);
    dim3 gridSize((halfsize+blockSize.x-1) / blockSize.x, (halfsize+blockSize.y-1) / blockSize.y);

    //sumKernel<<<(size+255) / 256, blockSize>>>(A_d, ans_d, n);
    sumKernel<<<gridSize, blockSize>>>(A_d, ans_d, n);

    cudaMemcpy(&ans, ans_d, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(ans_d);
}


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
