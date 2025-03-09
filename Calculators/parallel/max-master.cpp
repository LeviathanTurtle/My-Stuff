/* FINDING MAX VALUE USING PARALLEL
 * William Wadsworth
 * Created: at some point
 * Doctored: 5.25.2024 (merged 3 versions)
 * CSC-4310
 * 
 * This program finds the maximum value in an array serially and using two approaches using OpenMP
 * (parallel processing). The time to do all three is shown at the end.
 * TODO : maybe change approach, add -p switch (./max <array size> [-p] [num threads])
 * 
 * To compile:
 *     g++ max-master.cpp -Wall -fopenmp -o max
 * 
 * To run:
 *     ./max-master <array size> <thread count>
*/


#include <iostream>
#include <climits>   // INT_MIN
#include <omp.h>

using namespace std;

void parseArguments(const int, char**, int&, int&);
void fillArray(int*, const int&);
int findMax(const int*, const int&);
int findMaxParallel(const int*, const int&, const int&);
int findMaxReduction(const int*, const int&, const int&);

int main(int argc, char** argv)
{
    int N, T;
    parseArguments(argc, argv, N, T);

    // dynamically allocate nums array, fill with random values specified by user using argv
    int* array = new int[N];
    fillArray(array, N);

    // --- SERIAL -------------------------------
    // start time
    double s_start = omp_get_wtime();
    // find max value
    int s_max_value = findMax(array, N);
    // stop time
    double s_end = omp_get_wtime();

    // --- PARALLEL -----------------------------
    // start time
    double p_start = omp_get_wtime();
    // find max value
    int p_max_value = findMaxParallel(array, N, T);
    // stop time
    double p_end = omp_get_wtime();

    // --- PARALLEL - REDUCTION -----------------
    // start time
    double pr_start = omp_get_wtime();
    // find max value
    int pr_max_value = findMaxReduction(array, N, T);
    // stop time
    double pr_end = omp_get_wtime();

    delete[] array;

    cout << "The max is " << s_max_value << ", found serially in " << s_end-s_start << "s" << endl;
    cout << "The max is " << p_max_value << ", found in parallel in " << p_end-p_start << "s" << endl;
    cout << "The max is " << pr_max_value << ", found in parallel with reduction in " << pr_end-pr_start << "s" << endl;
    cout << endl;

    return 0;
}


void parseArguments(const int argc, char** argv, int& N, int& T)
{
    // check if I/O redirection is used correctly
    if (argc != 3) {
        cerr << "Usage: ./max-reduction <array size> <thread count>" << endl;
        exit(1);
    }

    // convert CLI arguments to ints
    N = atoi(argv[1]);
    // thread num
    T = atoi(argv[2]);
}


void fillArray(int* array, const int& N)
{
    srand(time(NULL)); // seed RNG

    for (int i=0; i<N; i++)
        array[i] = rand() % 1000;
}


int findMax(const int* array, const int& N)
{
    int max_val = INT_MIN;

    for (int i=0; i<N; i++)
        //if(array[i] > ans)
        //    ans = array[i];
        max_val = max(max_val, array[i]);

    return max_val;
}


int findMaxParallel(const int* array, const int& N, const int& T)
{
    int global_max = INT_MIN;

    if (N % T != 0) {
        cerr << "Error: array size N (" << N << ") must be perfectly divisible by the number of threads T (" << T << ")" << endl;
        delete[] array;
        return 1;
    }

    #pragma omp parallel num_threads(T)
    {
        int myrank = omp_get_thread_num();
        //cout << myrank << endl;

        // manually calculate loop boundaries
        int size = N/T;
        int j = myrank * size;
        int max_val = INT_MIN;
        
        //#pragma omp for
        for(int i=j; i<j+size; i++)
            //if(array[i] > max_val)
            //    max_val = array[i];
            max_val = max(max_val, array[i]);

        #pragma omp critical
        {
            //if(max_val > global_max)
            //    global_max = max_val;
            global_max = max(global_max, max_val);
        }
    }

    return global_max;
}


int findMaxReduction(const int* array, const int& N, const int& T)
{
    int max_val = INT_MIN;

    #pragma omp parallel for reduction(max:max_val) num_threads(T)
    for (int i=0; i<N; i++)
        //if(array[i] > ans)
        //    ans = array[i];
        max_val = max(max_val, array[i]);

    return max_val;
}
