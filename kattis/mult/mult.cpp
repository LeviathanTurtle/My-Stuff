#include <iostream>
using namespace std;

int main()
{
    // read in number of lines
    int numLines;
    cin >> numLines;

    int read, target, i=0;
    cin >> target >> read;

    while(i<numLines)
    {
        switch(read%target)
        {
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
