/* FINDING MAX VALUE USING PARALLEL
 * William Wadsworth
 * CSC-4310
 * ~/csc4310/parallel-max/max.cpp
 * 
 * This program 
 * 
 * To compile:
 * g++ max-serial.cpp -Wall -fopenmp -o max-serial
 * 
 * To run:
 * ./max [array size]
 * 
*/


#include <iostream>
#include <omp.h>

using namespace std;

int main(int argc, char** argv)
{
    // check if I/O redirection is used correctly (detailed in opening comment)
    if(argc != 2) {
        cout << "error: must have 3 arguments, " << argc << " provided. check opening comment.\n" << endl;
        return 1;
    }

    // convert CLI arguments to ints -- from ChatGPT
    // except it used stoi - I believe you mentioned atoi
    // array size
    int N = atoi(argv[1]);

    // dynamically allocate nums array, fill with random values
    // specified by user using argv
    int* array = new int[N];
    for(int i=0; i<N; i++)
        array[i] = rand()%1000;
    

    // global 
    int gmax = 0;

    double start = omp_get_wtime();
    for(int i=0; i<N; i++)
        if(array[i] > gmax)
            gmax = array[i];
    double end = omp_get_wtime();

    cout << "The max is " << gmax << ", found in " << end-start << endl;

    delete[] array;
    return 0;
}


