
#include <iostream>
using namespace std;

void partition(int** A, int dim)
{
    // create temp matrix T
    //int T[dim][dim];
    int** T = new int*[dim];
    for(int i=0; i<dim; i++)
        T[i] = new int[dim];

    for(int i=0; i<dim; i+=2)
        for(int j=0; j<dim; j+=2) {
            cout << "submatrix at (" << i << ", " << j << "):" << endl;
            for (int k = i; k < i + 2; k++) {
                for (int l = j; l < j + 2; l++) {
                    if (k < dim && l < dim) {
                        cout << A[k][l] << " ";
                    } else {
                         cout << "0 ";  // Zero padding for incomplete submatrices
                    }
                }
                cout << endl;
            }
        }
}


void mult(int** C, int** A, int** B, int dim)
{
    if(dim == 1) {
        C[0][0] = A[0][0] * B[0][0];
        //cout << C[0][0] << " ";
    }
    else {
        // create temp matrix T
        //int T[dim][dim];
        int** T = new int*[dim];
        for(int i=0; i<dim; i++)
            T[i] = new int[dim];

        // submatrix arrays
        int newSize = dim/2;
        // this is ugly I hate this
        int** subA1_1 = new int*[newSize];
        int** subA1_2 = new int*[newSize];
        int** subA2_1 = new int*[newSize];
        int** subA2_2 = new int*[newSize];

        int** subB1_1 = new int*[newSize];
        int** subB1_2 = new int*[newSize];
        int** subB2_1 = new int*[newSize];
        int** subB2_2 = new int*[newSize];

        int** subC1_1 = new int*[newSize];
        int** subC1_2 = new int*[newSize];
        int** subC2_1 = new int*[newSize];
        int** subC2_2 = new int*[newSize];

        int** subT1_1 = new int*[newSize];
        int** subT1_2 = new int*[newSize];
        int** subT2_1 = new int*[newSize];
        int** subT2_2 = new int*[newSize];
        for(int i=0; i<newSize; i++) {
            subA1_1[i] = new int[newSize];
            subA1_2[i] = new int[newSize];
            subA2_1[i] = new int[newSize];
            subA2_2[i] = new int[newSize];

            subB1_1[i] = new int[newSize];
            subB1_2[i] = new int[newSize];
            subB2_1[i] = new int[newSize];
            subB2_2[i] = new int[newSize];

            subC1_1[i] = new int[newSize];
            subC1_2[i] = new int[newSize];
            subC2_1[i] = new int[newSize];
            subC2_2[i] = new int[newSize];

            subT1_1[i] = new int[newSize];
            subT1_2[i] = new int[newSize];
            subT2_1[i] = new int[newSize];
            subT2_2[i] = new int[newSize];
        }
        // I can't wait for you to tell me there's an easier solution/way to do
        // this

        // partitioning algorithm -- from ChatGPT
        // modified to update array instead of output
        for(int i=0; i<newSize; i++)
            for(int j=0; j<newSize; j++) {
                //cout << "dim/2 at (" << i << ", " << j << "):" << endl;
                subA1_1[i][j] = A[i][j];
                subA1_2[i][j] = A[i][j+newSize];
                // here v
                subA2_1[i][j] = A[i+newSize][j];
                //      ^
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

        /*
        // add T to C
        for(int i=0; i<dim; i++)
            for(int j=0; j<dim; j++)
                C[i][j] += T[i][j];
        */
        for(int i=0; i<newSize; i++)
            for(int j=0; j<newSize; j++) {
                C[i][j] += T[i][j];
                C[i][j+newSize] += T[i][j+newSize];
                C[i+newSize][j] += T[i+newSize][j];
                C[i+newSize][j+newSize] += T[i+newSize][j+newSize];
            }

        // deallocation
        for(int i=0; i<newSize; i++) {
            delete[] subA1_1[i];
            delete[] subA1_2[i];
            delete[] subA2_1[i];
            delete[] subA2_2[i];

            delete[] subB1_1[i];
            delete[] subB1_2[i];
            delete[] subB2_1[i];
            delete[] subB2_2[i];

            delete[] subC1_1[i];
            delete[] subC1_2[i];
            delete[] subC2_1[i];
            delete[] subC2_2[i];

            delete[] subT1_1[i];
            delete[] subT1_2[i];
            delete[] subT2_1[i];
            delete[] subT2_2[i];
        }
        delete[] subA1_1;
        delete[] subA1_2;
        delete[] subA2_1;
        delete[] subA2_2;

        delete[] subB1_1;
        delete[] subB1_2;
        delete[] subB2_1;
        delete[] subB2_2;

        delete[] subC1_1;
        delete[] subC1_2;
        delete[] subC2_1;
        delete[] subC2_2;

        delete[] subT1_1;
        delete[] subT1_2;
        delete[] subT2_1;
        delete[] subT2_2;
        
        for(int i=0; i<dim; i++) {
            delete[] A[i];
            delete[] B[i];
            delete[] C[i];
            delete[] T[i];
        }
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] T;
    }
}


void printArray(int** array, int dim)
{
    for(int i=0; i<dim; i++) {
        for(int j=0; j<dim; j++)
            cout << array[i][j] << " ";
        cout << endl;
    }
}


int main()
{
    // --------------------------------------------------------------
    //  PARTITION TEST
    
    const int dim = 4;
    int cnt = 1;

    
    int** a = new int*[dim];
    for(int i=0; i<dim; i++)
        a[i] = new int[dim];

    int** b = new int*[dim];
    for(int i=0; i<dim; i++)
        b[i] = new int[dim];

    int** c = new int*[dim];
    for(int i=0; i<dim; i++)
        c[i] = new int[dim];



    for(int i=0; i<dim; i++)
        for(int j=0; j<dim; j++) {
            a[i][j] = cnt;
            cnt++;
        }
    cout << "A:\n";
    printArray(a,dim);
    cout << endl;

    for(int i=0; i<dim; i++)
        for(int j=0; j<dim; j++) {
            b[i][j] = cnt;
            cnt++;
        }
    cout << "B:\n";
    printArray(b,dim);
    cout << endl << endl;
    
    
    //int a[4][4] = {{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    //int b[4][4] = {{17,18,19,20},{21,22,23,24},{25,26,27,28},{29,30,31,32}};
    //int c[4][4] = {0};

    //partition(a,dim);
    

    mult(a,b,c,dim);


    // free memory, close program
    for(int i=0; i<dim; i++)
        delete[] a[i];
    return 0;
}