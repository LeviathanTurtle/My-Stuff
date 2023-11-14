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
 * 10    <-- number of lines
 * 8     <-- target value
 * 3
 * 12
 * 6
 * 24    <-- multiple of target
 * 14    <-- new target
 * 12
 * 9
 * 70    <-- multiple of target
 * 5     <-- new target
 *
 * Note: the arrows are not part of the file, they are just there so the 
 *       problem is easier to understand
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
    const int numLines;
    cin >> numLines;

    // temp variable for reading input
    int read;
    // variable for the target value
    int target
    // loop control variable
    int i=0;

    // read the target value and first temp value
    cin >> target >> read;

    // repeat for each line
    while(i<numLines) {
        // if the remainder of dividing the two numbers is 0, it is a multiple
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
