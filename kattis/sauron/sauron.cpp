/* 
 * William Wadsworth
 * Created: 
 * CSC - KATTIS
 *
 *
 * [DESCRIPTION]:
 * This program reads from a file using I/O redirection. The file contains a
 * string of vertical bars and parenthesis. If there are an equal number of 
 * vertical bars on either side of the parenthesis, "correct" is output. 
 * Otherwise, "fix" is output.
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
 * Input consists of a single string of length n, where 4 <= n <= 100. Input
 * strings will consist only of three types of characters: vertical bars, open
 * parentheses, and closing parentheses. Input strings contain one or more
 * vertical bars followed by a set of matching parentheses (the “eye”),
 * followed by one or more vertical bars. For a drawing to be “correct”, the
 * number of vertical bars on either side of the “eye” must match. Input will
 * always contain a pair of correctly matched parentheses, with no characters
 * between them. No other characters will appear in the string.
 * 
 * 
 * [DATA FILE EXAMPLE]:
 * |()||
 *
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
*/

#include <iostream>
using namespace std;

int main()
{
    // temp variable for current char value
    char item;
    // variables vertical bar counts on the left and right
    int lbarcnt=0, rbarcnt=0;
    // variable for marking when to count the right-side bars
    bool rparen=false;

    // repeat while there is input
    while(cin >> item) {
        switch(item) {
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
        cout << "correct\n"; // equal number of bars on either side
    else
        cout << "fix\n"; // unequal number of bars on either side

    return 0;
}
