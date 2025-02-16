#include <iostream>
using namespace std;

int main()
{
    int w;      // width of room
    int p;      // partitions
    int l[p+2];   // array of the partitions

    cin >> w >> p;
    l[0]=0;
    int i=1;
    for(i; i<p+1; i++)
        cin >> l[i];
    l[p+1]=w;

    for(int j=0; j<p+2; j++)
        cout << l[j] << " ";
    cout << endl;

    int t=1;


    return 0;
}
