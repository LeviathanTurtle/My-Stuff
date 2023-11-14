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
 *     g++ sauron.cpp -Wall -o sauron
 * 
 * To run:
 *     ./sauron < <data file>
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
    int lbarcnt=0, rbarcnt=0;
    char item;
    bool rparen=false;

    while(cin >> item)
    {
        switch(item)
        {
            case '(':
                break;
            case ')':
                rparen=true;
                break;
            
            default:
                if(rparen)
                    rbarcnt++;
                else
                    lbarcnt++;
                break;
        }
    }
    if(lbarcnt==rbarcnt)
        cout << "correct\n";
    else
        cout << "fix\n";

    return 0;
}
