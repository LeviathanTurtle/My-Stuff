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
 *     g++ nums.cpp -Wall -o nums
 * 
 * To run:
 *     ./nums < <data file>
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
    // temp variable for next/current number
    int read;
    // variable for keeping track of the next number
    int num=1;
    // variable for amount of numbers missed
    int cnt=0;
    // loop control
    int i=0;

    // read in amount of numbers in file
    cin >> read;
    int length = read;

    // read in first number
    cin >> read;
    // repeat for each remaining number in file
    while(i<length) {
        if(read==num) {
            // number is present
            cin >> read;
            num++;
            i++;
        }
        else {
            // number is missing
            cout << num << endl;
            cnt++;
            num++;
        }
    }

    // if no numbers were missed
    if(cnt==0)
        cout << "good job" << endl;

    return 0;
}
