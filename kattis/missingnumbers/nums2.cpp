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
