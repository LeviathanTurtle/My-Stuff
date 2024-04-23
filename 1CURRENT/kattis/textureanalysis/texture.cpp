#include <iostream>
#include <string>
using namespace std;

#define MAX_LEN 2000

int main()
{
    int i=1;    // line counter
    char pixel; // read-in variable

    int wpixelcnt=0;    // white pixel count
    int targetcnt=0;    // target white pixel count
    bool isEven=true;   // assume even, override if not
    bool first=true;    // check if first in line

    //pixel=getchar();
    //cin >> pixel;
    cin.get(pixel);
    while(pixel!='E')
    {
        switch(pixel)
        {
            case '*':
                
                if(first)
                    first=false;
                break;
            
            case '\n':
                cout << i << " ";
                if(targetcnt==wpixelcnt)
                    cout << "EVEN\n";
                else
                    cout << "NOT EVEN\n";
                i++;
                break;
            
            default:
                if(first)
                    targetcnt++;
                else
                    wpixelcnt++;
        }
        cin.get(pixel);
    }

    return 0;
}
