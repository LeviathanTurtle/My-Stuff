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

    int card;                   // read value
    int *sums;                  // dynamic array for sums
    sums = new int[numCards]();
    double largest=0;           // keep track of largest value

/*
    // DEBUG
    cout << "empty array (temp):\n";
    for(int i=0; i<numCards; i++)
        cout << temp[i] << " ";
    cout << endl << "empty array (sums):\n";
    for(int i=0; i<numCards; i++)
        cout << sums[i] << " ";
    cout << endl << endl;
*/

    // THE MEAT OF THE SANDWICH
    int i=1;        // loop control
    int g;          // i value
    
    while(cin >> card)
    {
        // update sums array
        for(int h=0; h<i; h++) {
            sums[h]+=card;
            if(sums[h]>largest) {
                largest=sums[h];
                g=i;
            }
        }
        i++;
    }

/*
    // DEBUG
    cout << "constructed array (temp):\n";
    for(int i=0; i<numCards; i++)
        cout << temp[i] << " ";
    cout << endl << "constructed array (sums):\n";
    for(int i=0; i<numCards; i++)
        cout << sums[i] << " ";
    cout << endl;
*/
    cout << largest/g << endl;
    delete [] sums;
    return 0;
}


