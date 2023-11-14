/* 
 * William Wadsworth
 * Created: 
 * CSC - KATTIS
 *
 *
 * [DESCRIPTION]:
 * This program reads from a data file sets the second read value to be the
 * "target" value. If a following number is a multiple of the target, then it
 * is output. The following value becomes the next target value.
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
 * The first line of input contains an integer n (2 <= n <= 1000), the length
 * of the number sequence. The following lines contains the sequence, one
 * number per line. All numbers in the sequence are positive integers <= 100.
 * The sequence is guaranteed to contain at least one complete round of the
 * game (but may end with an incomplete round).
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
