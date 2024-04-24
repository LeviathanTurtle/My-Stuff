/* ADDING VECTORS USING CUDA
 * William Wadsworth
 * 10.18.2023
 * CSC 4310
 * 
 * This program opens and reads two files containing vector data. Data files 
 * start with an integer n and follow with n floats. The program dynamically
 * allocates two matrices and fills them with the data from their respective
 * files. If the sizes of the two vectors A and B match, the program uses CUDA
 * to add the two vectors together, storing them in a third vector C. The
 * contents of C are then output to a file passed as an execution argument.
 * 
 * To compile:
 *     nvcc vectorAdd.cpp -o va
 * 
 * To run:
 *     ./va [vector file 1] [vector file 2] [output file]
*/

#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

using namespace std;

float* loadVector(const char*, int&);
void dump(const char*, float*, int);
void addVector(float*, float*, float*, int);


int main(int argc, char* argv[])
{
    // check if arguments were used correctly
    if(argc != 4) {
        cerr << "error: incorrect number of arguments\n";
        exit(1);
    }

    // create and load vectors, make note of size, will be checked later
    int sizeA, sizeB;
    float* A = loadVector(argv[1],sizeA);
    float* B = loadVector(argv[2],sizeB);
    float* C;

    if(sizeA == sizeB) {
        // initialize C, add A and B 
        // does not matter between sizeA and sizeB, they are equal
        C = new float [sizeA];
        addVector(A,B,C,sizeA);
    }
    else {
        cerr << "error: vector sizes are not the same\n";
        exit(666);
    }

    // write contents of output vector to file
    dump(argv[3],C,sizeA); // does not matter between sizeA and sizeB, they are equal

    // free memory
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}


// this function takes a filename as an arg to open (const char to avoid 
// accidentally editing it). The function then reads the size of the float
// specified in the file, and dynamically creates a new float array (vector)
// and loads it with the data from the file. the newly created float vector is 
// returned. vectorSize is included and passed by reference so we can access 
// the vector's size from main
float* loadVector(const char* fileName, int& vectorSize)
{
    // set up input file
    ifstream inputFile(fileName);

    // if inpute file was not able to be opened, error and exit
    if(!inputFile) {
        cerr << "error: file unable to be opened\n";
        exit(1);
    }

    // read in vector size
    int size;
    inputFile >> size;
    // allow main to access this vector's size
    vectorSize = size;
    // validate vector size, if invalid, error and exit
    if(size <= 0) {
        cerr << "error: vector size invalid, must be greater than 0\n";
        exit(2);
    }

    // dynamically create new vector based on size from file
    float* vector = new float [size];

    // load vector with data from file
    for(int i=0; i<size; i++)
        inputFile >> vector[i];


    inputFile.close();
    return vector;
}


// this function takes a filename as an arg to open (const char to avoid
// accidentally editing it). The function attenpts to open said file, and 
// outputs an error if unsuccessful. Otherwise, it first outputs the size of
// the vector that is passed, followed by the vector data separated by a space.
void dump(const char* fileName, float* vector, int size)
{
    // output file
    ofstream outputFile(fileName);

    // if output file was not able to be opened, error and exit
    if(!outputFile) {
        cerr << "error: file unable to be opened\n";
        exit(1);
    }

    // output the size to maintain consistent vector file structure
    outputFile << size << " ";
    // write each vector element to file
    for(int i=0; i<size; i++)
        outputFile << vector[i] << " ";

    // close file
    outputFile.close();
}


__global__
void addVectorKernel(float* A, float* B, float* C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n)
        C[i] = A[i] + B[i];
}


void addVector(float* A, float* B, float* C, int n)
{
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    addVectorKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

