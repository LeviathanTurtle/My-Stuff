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
    int read, num=1, i=0, cnt=0;
    cin >> read;
    int length = read;

    cin >> read;
    while(i<length)
    {
        if(read==num) // number is present
        {
            cin >> read;
            num++;
            i++;
        }
        else // number is missing
        {
            cout << num << endl;
            cnt++;
            num++;
        }
    }
    if(cnt==0)
        cout << "good job" << endl;

    return 0;
}
