#include <iostream>
#include <iomanip>
using namespace std;

int main()
{
    // read in number of cards
    int numCards;
    cin >> numCards;
    // set output to show 9 decimals
    cout << fixed << showpoint << setprecision(9);

    // convert numCards to const to define array size
    int card, matrix[numCards][numCards]={};
/*
    // DEBUG
    cout << "empty array:\n";
    for(int i=0; i<numCards; i++) {
         for(int j=0; j<numCards; j++)
             cout << matrix[i][j] << " ";
         cout << endl;
    }
    cout << endl;
*/

    // read in first card, top left in matrix
    cin >> card;
    matrix[0][0]=card;

    // MATRIX
    cin >> card;
    for(int i=1; i<numCards; i++) {
        for(int j=0; j<=i; j++)
            matrix[i][j]=card+matrix[i-1][j];
        cin >> card;
    }

    // FIND LARGEST
    int e;
    double largest=0;
    for(int i=0; i<numCards; i++)
        for(int j=0; j<numCards; j++)
            if(matrix[i][j]>largest) {
                largest=matrix[i][j];
                e=i+1;
            }
    if(largest<0)
        cout << "0";
    else
        cout << largest/e << endl;
    
/*
    // DEBUG OUTPUT
    cout << "constructed array:\n";
    for(int i=0; i<numCards; i++) {
        for(int j=0; j<numCards; j++)
            cout << matrix[i][j] << " ";
        cout << endl;
    }
*/
    return 0;
}


