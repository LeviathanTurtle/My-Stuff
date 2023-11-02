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
