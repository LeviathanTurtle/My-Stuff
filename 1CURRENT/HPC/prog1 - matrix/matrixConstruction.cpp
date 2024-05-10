#include <iostream>
#include <fstream>

using namespace std;

int main()
{
    //  INTRODUCTION
    cout << "This function creates two matrices of user-defined size.\n\n";
    
    //  MATRIX DIMENSIONS
    int dim1, dim2, dim3, max;
    cout << "matrix1 row dim: ";
    cin >> dim1;
    cout << "matrix1 col dim: ";
    cin >> dim2;
    cout << "matrix2 col dim: ";
    cin >> dim3;
    cout << "max value: ";
    cin >> max;

    //  USER NAMES FILES
    string matrixfile1, matrixfile2;
    cout << "Name first matrix file: ";
    cin >> matrixfile1;
    cout << "Name second matrix file: ";
    cin >> matrixfile2;

    //  CONFIRMATION
    cout << "\nYou have chosen to construct:\n" << "matrix1: " << dim1 << "x" << dim2;
    cout << "\nmatrix2: " << dim2 << "x" << dim3;
    cout << "\nmax value: " << max;
    cout << "\nmatrix1 filename: " << matrixfile1;
    cout << "\nmatrix2 filename: " << matrixfile2;

    cout << "\n\nConfirm [Y/n]: ";
    char conf;
    cin >> conf;

    //  MATRIX CONSTRUCTION
    ofstream outfile1 (matrixfile1);
    ofstream outfile2 (matrixfile2);
    if(conf == 'Y') {
        cout << "Processing...\n";
        // write to respective files
        outfile1 << dim1 << " " << dim2 << endl;
        for(int i=0; i<dim1; i++) {
            for(int j=0; j<dim2; j++)
                outfile1 << rand()%max << " ";
            outfile1 << endl;
        }
        outfile1.close();

        outfile2 << dim2 << " " << dim3 << endl;
        for(int i=0; i<dim2; i++) {
            for(int j=0; j<dim3; j++)
                outfile2 << rand()%max << " ";
            outfile2 << endl;
        }

        outfile2.close();
        cout << "Done.\n";
    }
    else {
        cout << "You are choosing to restart the function. Restarting...\n\n";
        main();
    }

    return 0;
}