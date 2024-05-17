/* P3 - MATRIX MULTIPLICATION USING CUDA WITH SHARED MEMORY
 * William Wadsworth
 * Created: 11.19.2023
 * 
 * CSC 4310
 * ~/csc4310/cuda/p3/p3-cudaMult-a.cu
 * 
 * 
 * [DESCRIPTION]:
 * This program uses one thread to compute each element of a product matrix. To
 * maximize the size of the matrices used in the multiplication, the matrix is
 * tiled (blocks are set up on the grid). This program does not use shared
 * memory. 3 times are output: time elapsed in main function, time elapsed in
 * cuda setup function (including kernel), time elapsed in just the kernel. 
 * This program utilizes shared memory.
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     nvcc p3-cudaMult-a.cu -o p3-a
 * 
 * To run (5 args):
 *     ./p3-a <tile width> <matrix A file> <matrix B file> <output file>
 * 
 * <tile width>    - can be: 4, 8, 16, 32
 * <output file>   - file that will contain the output of this program, follows
 *                   matrix file format below.
 * 
 * <matrix A file>,
 * <matrix B file> - square matrices, first integer is the size N, the next N
 *                   lines have N floats.
 * Ex:
 * 4
 * 1 2 3 4
 * 2 3 4 5
 * 3 4 5 6
 * 4 5 6 7
 * 
 * Note: using integers here so it is easier to read.
 * 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - CLI args used incorrectly
 * 
 * 2 - file unable to be created or opened.
 * 
 * 3 - CUDA error
 * 
 * 4 - invalid tile width (can be 4, 8, 16, or 32)
 * 
 * 5 - matrix dimension is not a multiple of the tile width
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;


// function prototypes
// function to allocate a 1D matrix
float* loadMatrix(char*, int&);
// function to output 1D matrix
void dump(char*, float*, const int&);
// function to error check CUDA functions
void checkCudaError(cudaError_t, const char*);
// CUDA kernel, matrix multiply
__global__ void multKernel(float*, float*, float*, int, int);
// function to set up CUDA kernel for matrix multiply
void mult(float*, float*, float*, int, char*);


int main(int argc, char* argv[])
{
    // timing for items in main function
    struct timeval main_startTime, main_stopTime;
    double main_start, main_stop;
    // timing for cuda setup + kernel in mult() function

    // start time
    gettimeofday(&main_startTime,NULL);
    
    // check CLI args
    if(argc != 5) {
        cerr << "error: CLI args used incorrectly.\nCorrect execution: ./p1 "
             << "<tile width> <matrix A file> <matrix B file> <output file>.\n";
        exit(1);
    }

    // variable for size, defined here so it can be passed to functions
    int size;

    // load matrices
    float* a = loadMatrix(argv[2],size);
    float* b = loadMatrix(argv[3],size);
    float* c = new float [size*size];
    // THIS ASSUMES THE MATRICES ARE THE SAME SIZE

    mult(a,b,c,size,argv[1]);
 
    // output function
    dump(argv[4], c, size);

    // deallocate memory
    delete[] a;
    delete[] b;
    delete[] c;

    // timing for items in main function
    gettimeofday(&main_stopTime,NULL);
    
    // calculate time elapsed in main
    main_start = main_startTime.tv_sec + (main_startTime.tv_usec/1000000.0);
    main_stop = main_stopTime.tv_sec + (main_stopTime.tv_usec/1000000.0); 

    // output time
    cout << "main function time: " << main_stop-main_start << endl;

    return 0;
}


/* pre-condition: filename is defined in argv and size is declared in main
 *
 * post-condition: dynamic 1D array is initialized, loaded, and returned to
 *                 main
*/
float* loadMatrix(char* filename, int& size)
{
    // create/open file
    ifstream inputFile (filename);

    // check if file was opened
    if(!inputFile) {
        cerr << "error: file unable to be created or opened (provided name: "
             << filename << ").\n";
        exit(2);
    }

    // get size from file
    inputFile >> size;

    // dynamically allocate matrix 
    float* matrix = new float [size*size];    

    // read data from input file
    for(int i=0; i<size*size; i++)
        inputFile >> matrix[i];

    // close file
    inputFile.close();
    return matrix;
}


/* pre-condition: filename is defined in argv, matrix is declared and loaded
 *                with data, and size is declared and initialized
 *
 * post-condition: nothing is returned, the array is output to a file
*/
void dump(char* filename, float* matrix, const int& size)
{
    // create/open file
    ofstream outputFile (filename);

    // check if file was opened
    if(!outputFile) {
        cerr << "error: file unable to be created or opened (provided name: "
             << filename << ").\n";
        exit(2);
    }

    // output size 
    outputFile << size << endl;

    // set output specifications -- %6.2f
    outputFile << fixed << setprecision(2);

    // output matrix data
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++)
            outputFile << fixed << setprecision(2) << setw(6) << matrix[i*size+j] << " ";
        outputFile << endl;
    }

    // close file
    outputFile.close();
}


/* pre-condition: error, message are returned from a CUDA function call
 * 
 * post-condition: if CUDA function call was unsuccessful, it prints an error
 *                 message and terminates the program
*/
void checkCudaError(cudaError_t error, const char* message)
{
    if(error != cudaSuccess) {
        cerr << "Error in " << message << ": " << cudaGetErrorString(error) << endl;
        exit(3);
    }
}


/* pre-condition: matrices a and b are defined and loaded with data, matrix c
 *                is declared, and size is initialized to a value from the 
 *                matrix files
 *
 * post-condition: nothing is returned, the function calculates a product and 
 *                 stores it in the product array
*/
__global__ void multKernel(float* A, float* B, float* product, int size, int tileWidth)
{
    extern __shared__ float Ads_Bds[];
    // load M and N into shared memory
    float* Ads = Ads_Bds;
    //float *Bds = (float *) Ads_Bds + Ads_sz;
    float* Bds = &Ads_Bds[tileWidth*tileWidth];

    // thread/block variables
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Calculate the row index of the P element and M
    int Row = by * tileWidth + ty;
    // Calculate the column index of P and N
    int Col = bx * tileWidth + tx;

    float pValue = 0;

    // 
    float A_val = A[Row * size + tx];
    float B_val = B[ty * size + Col];

    //for(int i=0; i<size/tileWidth; i+=2) {
    for(int i=1; i<=size/tileWidth; i++) {
        // copy to shared memory
        Ads[ty * tileWidth + tx] = A_val;
        //Ads[ty * tileWidth + tx] = A[Row * size + tx];
        Bds[ty * tileWidth + tx] = B_val;
        //Bds[ty * tileWidth + tx] = B[ty * size + Col];

        __syncthreads();

        // update shared memory index
        A_val = A[Row*size + i*tileWidth + tx];
        //Ads[ty * tileWidth + tx] = A[Row*size + i*tileWidth + tx];
        B_val = B[(i*tileWidth + ty)*size + Col];
        //Bds[ty * tileWidth + tx] = B[(i*tileWidth + ty)*size + Col];

        // Loop unrolling
        #pragma unroll
        // main computation loop
        for (int j=0; j<tileWidth; j+=4) {
            pValue += Ads[ty * tileWidth + j] * Bds[j * tileWidth + tx];
            pValue += Ads[ty * tileWidth + j + 1] * Bds[(j + 1) * tileWidth + tx];
            pValue += Ads[ty * tileWidth + j + 2] * Bds[(j + 2) * tileWidth + tx];
            pValue += Ads[ty * tileWidth + j + 3] * Bds[(j + 3) * tileWidth + tx];
        }

        __syncthreads();
    }
    // update product
    product[Row * size + Col] = pValue;
}


/* pre-condition: matrices a b and c are declared and a and b are loaded with
 *                data, tile is defined in argv
 *
 * post-condition: nothing is returned, the function copies the contents of the
 *                 temp product array into c from main
*/
void mult(float* a, float* b, float* c, int n, char* tile)
{
    // total cuda time variables
    struct timeval cuda_startTime, cuda_stopTime;
    double cuda_start, cuda_stop;
    // cuda KERNEL timing
    struct timeval kernel_startTime, kernel_stopTime;
    double kernel_start, kernel_stop;

    // start clock
    gettimeofday(&cuda_startTime,NULL);
    
    // matrices for cuda
    float *a_d, *b_d, *c_d;
    // calculate 1D array size
    int size = n * n * sizeof(float);

    // get tile width from argv
    int tileWidth = atoi(tile);
    //cout << "n: " << n << ", tileWidth: " << tileWidth << endl;

    // check for valid tile width
    if(tileWidth < 4 || tileWidth > 32) {
        cerr << "error: invalid tile width. must be: 4, 8, 16, or 32.\n";
        exit(4);
    }

    // check if the number of tiles overlaps the size of the matrix
    // this is to avoid divergence in the tile that processes the remaining 
    // portion
    if(n%tileWidth != 0 && tileWidth%n != 0) {
        cerr << "error: matrix dimension must be a multiple of the tile width.\n";
        exit(5);
    }

    // calculate number of tiles based on matrix size
    int numTiles = (n+tileWidth-1) / tileWidth;

    // allocate memory for new arrays
    checkCudaError(cudaMalloc((void **) &a_d, size), "cudaMalloc for a_d");
    checkCudaError(cudaMalloc((void **) &b_d, size), "cudaMalloc for b_d");
    checkCudaError(cudaMalloc((void **) &c_d, size), "cudaMalloc for c_d");

    // copy data into respective arrays
    checkCudaError(cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice), "cudaMemcpy a_d");
    checkCudaError(cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice), "cudaMemcpy b_d");

    // define grid and block size -- SQUARE
    dim3 gridsize(numTiles, numTiles, 1);
    dim3 blocksize(tileWidth, tileWidth, 1);

    // calculate size, will be used for shared memory
    size_t sharedMemSize = 2 * tileWidth * tileWidth * sizeof(float);
    cudaFuncSetAttribute(multKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);

    // cuda kernel
    gettimeofday(&kernel_startTime,NULL);
    multKernel<<<gridsize, blocksize,sharedMemSize>>>(a_d, b_d, c_d, n, tileWidth);
    gettimeofday(&kernel_stopTime,NULL);

    // calculate just kernel time
    kernel_start = kernel_startTime.tv_sec + (kernel_startTime.tv_usec/1000000.0);
    kernel_stop = kernel_stopTime.tv_sec + (kernel_stopTime.tv_usec/1000000.0); 
    // output just cuda kernel time
    cout << "kernel function time: " << kernel_stop-kernel_start << endl;

    // copy results to matrix from main
    cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);

    // deallocate memory
    checkCudaError(cudaFree(a_d), "cudaFree for a_d");
    checkCudaError(cudaFree(b_d), "cudaFree for b_d");
    checkCudaError(cudaFree(c_d), "cudaFree for c_d");

    // stop total cuda clock
    gettimeofday(&cuda_stopTime,NULL);

    // calculate total cuda time
    cuda_start = cuda_startTime.tv_sec + (cuda_startTime.tv_usec/1000000.0);
    cuda_stop = cuda_stopTime.tv_sec + (cuda_stopTime.tv_usec/1000000.0);

    // output total cuda time
    cout << "cuda function time: " << cuda_stop-cuda_start << endl;
}

