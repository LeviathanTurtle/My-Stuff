#include <iostream>
#include <algorithm>
using namespace std;

int main()
{
    // read number of signatures
    int K;
    cin >> K;

    int passes=1;   // counter for number of passes
    int temp;       // read in number
    int i=0;        // loop control

    // load original list to retain order
    int original[K];
    cout << "original array: ";
    for(int i=0; i<K; i++)
    {
        cin >> temp;
        original[i]=temp;
        cout << original[i] << " ";
    }
    cout << endl;

    // sort list to determine which value is next
    int sorted[K];
    for(int i=0; i<K; i++)
        sorted[i]=original[i];
    sort(original, original+K+1);

    // DEBUG   
    cout << "sorted array: ";
    for(int i=0; i<K; i++)
        cout << original[i] << " ";
    cout << endl;


    int cnt=0;
    bool notDone=true;
    while(notDone)
    {
        for(int i=0; i<K; i++)
        {
            if(original[i]==sorted[i]) {
                original[i]=0;
                cout << original[i] << " ";
            }
            else {
                passes++;
                cout << "passes: " << passes << endl;
            }
        }

        // check if done
        for(int i=0; i<K; i++)
            if(original[i]==0)
                cnt++;
        if(cnt==K+1)
            notDone=false;
    }

    cout << passes << endl;
    return 0;
}
