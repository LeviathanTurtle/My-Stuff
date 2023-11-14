/* 
 * William Wadsworth
 * Created: 
 * CSC - KATTIS
 *
 *
 * [DESCRIPTION]:
 * This program reads from a data file. The first value is the amount of
 * numbers in the file, and the next lines count from 1 to 200. Any numbers
 * missing will be output.
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
 * The first line of input contains a single integer n, where 1 <= n <= 100.
 * Each of the next n lines contains one number that the child recited. Each
 * recited number is an integer between 1 and 200 (inclusive). They are listed
 * in increasing order, and there are no duplicates.
 * 
 * 
 * [DATA FILE EXAMPLE]:
 * 9   <-- number of values
 * 2   <-- 1 is missing, output 1
 * 4   <-- 3 is missing, output 3
 * 5
 * 7   <-- 6 is missing, output 6
 * 8
 * 9
 * 10
 * 11
 * 13  <-- 12 is missing, output 12
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
