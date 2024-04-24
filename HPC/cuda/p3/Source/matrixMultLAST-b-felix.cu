/*
    Name: Joshua Patterson
    Date: 11/06/23 
    Class: CSC-4310-01
    Location of program: jpatterson/csc4000/hw/matrixMult
    
    Description: This program does a matrix multiply with cuda that takes in a tile size and separates it through that. everything is timed in seconds using gettimeof day, all of the entries have been set to be output in the desired format, I have  also included cuda error handling for the sake of posterity
    
    compile code

    nvcc matrixMultLast.cu -o [any name you want to use, i used matrixMult in my testing]
    
    execute code
    
    ./matrixMultLast <tilewidth> <input.txt> <input2.txt> <output.txt> 
*/



#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <sys/time.h>

float* fileMaker(const char* filename, int &count);
void cudaSetup(float* A_hvi , float* B_h ,float* C_h, int n, int TILE_WIDTH);
void fileOutput(const char* filename, float* C, int n);
__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width, int TILE_WIDTH);
void checkCudaError(cudaError_t err, const char* msg);


using namespace std;


// main function
// Precondition: Must be provided with two input filenames for matrices and one output filename as command line arguments.
// Process: Initializes matrices, calls CUDA setup and matrix multiplication, outputs result to file, and releases memory.
// Postcondition: The product of the two input matrices is saved to the output file, and execution time is printed to the console.
int main(int argc, char *argv[] ) {

   struct timeval startOverall, endOverall;
   gettimeofday(&startOverall, NULL);

    if (argc < 5) {
        cout << "Usage: " << argv[0] << "<tile_width> <infile> <infile> <outfile>\n";
        cout << endl << "You have too few arguments."<< endl;
        return 1;
    }

    if (argc > 5) {
        cout << "Usage: " << argv[0] << "<tile_width> <infile> <infile> <outfile>\n";
        cout << endl << "You have too many arguments." << endl;
        return 1;
    }

    int count = 0;
    int countb = 0;
    int TILE_WIDTH = stoi(argv[1]);

    if (TILE_WIDTH >= 64) {
       cout << "Tile width selected is too large and won't work. Terminating program." << endl;
       return 1;
    }
    float *A_h = fileMaker(argv[2], count);
    float *B_h = fileMaker(argv[3], countb);

    if (count != countb) {
        cout << "N values are incompatible with each other\n";
        return 1;
    }

    float *C_h = (float *)calloc(count * count, sizeof(float));

    cudaSetup(A_h, B_h, C_h, count, TILE_WIDTH);

    fileOutput(argv[4], C_h, count);

    delete[] A_h;
    delete[] B_h;
    delete[] C_h;

    gettimeofday(&endOverall, NULL);
    double timeOverall = (endOverall.tv_sec - startOverall.tv_sec) * 1000.0; // sec to ms
    timeOverall += (endOverall.tv_usec - startOverall.tv_usec) / 1000.0; // us to ms
    cout << "Overall time: " << timeOverall << " ms" << endl;
    return 0;
}

// fileOutput function
// precondition: filename must be valid and C must be initialized
// during: Writes the content of array C to a file
// postcondition: Output matrix is saved to a file
void fileOutput(const char* filename, float* C, int n){

   ofstream outFile(filename);
   
   if (!outFile){
      cerr << "Error: Could not open output file " << filename << endl;
      exit(1);
   }
   
   if (!(outFile << n << endl)){
      cerr << "Error: Could not write to output file " << filename << endl;
      outFile.close();
      exit(1);
   }
 
   for(int i=0; i<n; i++){
      for(int j=0; j<n; j++){
         outFile << fixed << setprecision(2) << setw(6) << C[i*n+j] << ' ';
      }
      outFile << endl;
   } 

   
   outFile.close();
}

// fileMaker function
// precondition: filename must be valid
// during: Reads a vector from a file
// postcondition: Returns a dynamically allocated array and updates count
float* fileMaker(const char* filename, int &count){

   
   ifstream file(filename);
   
   if (!file){
      cerr << "File " << filename << " is invalid: Could not open file." << endl;
      exit(1);  
   }

   if (!(file >> count)){
      cerr << "File " << filename << " is invalid: Could not read 'count'." << endl;
      file.close();
      exit(1);
   }


   if (count%2!=0){
      cerr << "Count is not even and I have chosen not to allow that" << endl;
      exit(1);
   }
   

   float* vector = new float[count*count];


   if(count<=0){
      cerr<< "File " << filename << "is invalid: Count less than or equals to 0." << endl;
      exit(1);
   }
   
   for(int i = 0; i < count; i++) {
    for(int j = 0; j < count; j++) {
        if (!(file >> vector[i*count+j])) {
            cerr << "Error reading matrix values" << endl;
            exit(1);
        }
    }
}

   
   file.close();
   return vector;

}

// checkCudaError function
// precondition: cudaError_t err contains the status code returned from a CUDA_h A_hPI call, msg is a description of the A_hPI call.
// during: Checks if the CUDA_h A_hPI call specified by 'msg' was successful by examining 'err'.
// postcondition: If the A_hPI call was successful, function returns normally. Otherwise, prints an error message and terminates the program.
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "Error in " << msg << ": " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

// cudaSetup function
// Precondition: Host matrices A_h and B_h must be initialized and C_h must be allocated, n represents the size of these square matrices.
// Process: Allocates device memory, copies matrices from host to device, configures grid and block dimensions, executes the kernel, and copies the result back to the host.
// Postcondition: The matrix multiplication is performed on the GPU and the result is copied back to C_h on the host.
void cudaSetup(float* A_h, float* B_h ,float* C_h, int n, int TILE_WIDTH) {
   struct timeval startCuda, endCuda, startKernel, endKernel;
   gettimeofday(&startCuda, NULL);

   float *a_d, *b_d, *c_d;
   int size = n*n*sizeof(float);
   
   
   checkCudaError(cudaMalloc((void **) &a_d, size), "cudaMalloc for a_d");
   checkCudaError(cudaMalloc((void **) &b_d, size), "cudaMalloc for b_d");

   checkCudaError(cudaMallocManaged(&c_d, size), "cudaMallocManaged for c_d");
   checkCudaError(cudaMemset(c_d, 0, size), "cudaMemset for c_d");

   checkCudaError(cudaMemcpy(a_d, A_h, size, cudaMemcpyHostToDevice), "cudaMemcpy for a_d");
   checkCudaError(cudaMemcpy(b_d, B_h, size, cudaMemcpyHostToDevice), "cudaMemcpy for b_d");

   dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
   dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);
   gettimeofday(&startKernel, NULL);
   MatrixMulKernel<<<grid, block, 2*TILE_WIDTH*TILE_WIDTH*sizeof(float)>>>(a_d, b_d, c_d, n, TILE_WIDTH);
   checkCudaError(cudaGetLastError(), "executing kernel");
   gettimeofday(&endKernel, NULL);
   

   checkCudaError(cudaMemcpy(C_h, c_d, size, cudaMemcpyDeviceToHost), "cudaMemcpy for C");
   
   checkCudaError(cudaFree(a_d), "cudaFree for a_d");
   checkCudaError(cudaFree(b_d), "cudaFree for b_d");
   checkCudaError(cudaFree(c_d), "cudaFree for c_d");
   gettimeofday(&endCuda, NULL);
   double timeKernel = (endKernel.tv_sec - startKernel.tv_sec) * 1000.0;
   timeKernel += (endKernel.tv_usec - startKernel.tv_usec) / 1000.0;
   double timeCuda = (endCuda.tv_sec - startCuda.tv_sec) * 1000.0;
   timeCuda += (endCuda.tv_usec - startCuda.tv_usec) / 1000.0;

   cout << "Kernel time: " << timeKernel << " ms" << endl;
   cout << "CUDA operations time: " << timeCuda << " ms" << endl;
}

// MatrixMulKernel function (CUDA kernel)
// Precondition: Matrices M, N, and P must be allocated on the device, and Width must represent their dimensions.
// Process: Each thread calculates one element of the product matrix P by multiplying the corresponding row of M with the column of N.
// Postcondition: The product matrix P is fully computed on the device.
__global__
void MatrixMulKernel(float *M, float *N, float *P, int Width, int TILE_WIDTH){
 extern __shared__ float sharedMem[];
    float* subTileM = sharedMem;
    float* subTileN = &sharedMem[TILE_WIDTH*TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;
    float M_reg, N_reg;

    if (Row < Width && Col < Width) {
        M_reg = M[Row * Width + tx];
        N_reg = N[ty * Width + Col];
    }

    for (int m = 1; m <= Width/TILE_WIDTH; ++m) {
        if (Row < Width && Col < Width) {
            subTileM[ty * TILE_WIDTH + tx] = M_reg;
            subTileN[ty * TILE_WIDTH + tx] = N_reg;
        }
        __syncthreads();
        if (m < Width/TILE_WIDTH) {
            if (Row < Width && Col < Width) {
                M_reg = M[Row*Width + m*TILE_WIDTH + tx];
                N_reg = N[(m*TILE_WIDTH + ty)*Width + Col];
            }
        }
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += subTileM[ty * TILE_WIDTH + k] * subTileN[k * TILE_WIDTH + tx];

        __syncthreads();
    }
    if (Row < Width && Col < Width) {
        P[Row*Width + Col] = Pvalue;
    }
}


