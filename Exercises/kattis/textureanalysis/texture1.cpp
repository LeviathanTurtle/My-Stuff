#include <iostream>
#include <string>
#include <string.h>
#include <stdio.h>
#include <fstream>
using namespace std;

#define MAX_LEN 1000

int main()
{
    int i=1;    // line counter
//    int h=0;    // array control
//    char pixel; // read-in variable

    int wpixelcnt=0;    // white pixel count
    int targetcnt=0;    // target white pixel count
    int bcnt=0;
    bool first=true;

    string line;

    getline(cin,&line);
    cin.ignore();
    while(line[0]!='E')
    {
        if(first && line[i]=='.')
            targetcnt++;
        else if(line[i]=='*')
            bcnt++;
        else if(line[i]=='*' && bcnt==2)
            first=false;
        else
            wpixelcnt++;


        i++;
        getline(cin,&line);
        cin.ignore();
    }

    return 0;
}
