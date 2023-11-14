/* 
 * William Wadsworth
 * Created: 
 * CSC - KATTIS
 *
 *
 * [DESCRIPTION]:
 * This program 
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ mult.cpp -Wall -o mult
 * 
 * To run:
 *     ./mult < <data file>
 * 
 * 
 * [DATA FILE STRUCTURE]:
 * 
 * 
 * 
 * [DATA FILE EXAMPLE]:
 * 
 *
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
*/

#include <iostream>
using namespace std;

int main()
{
    // read in number of lines
    int numLines;
    cin >> numLines;

    int read, target, i=0;
    cin >> target >> read;

    while(i<numLines) {
        switch(read%target) {
            // mult is found
            case 0:
                cout << read << endl;
                cin >> target >> read;
                i+=2;
                break;

            // not found
            default:
                cin >> read;
                break;
        }
        i++;
    }

    return 0;
}
