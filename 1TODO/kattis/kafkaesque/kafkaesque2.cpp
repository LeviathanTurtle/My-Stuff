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
//    int i=0;        // loop control

    int original[K];    // load original list to retain order
    int sorted[K];      // sort list to determine which value is next

    cout << "original array: ";
    for(int i=0; i<K; i++)
    {
        cin >> temp;
        original[i]=temp;
        sorted[i]=temp;
        cout << original[i] << " ";
    }
    sort(sorted,sorted+K);
    cout << endl;



    // DEBUG   
    cout << "sorted array:   ";
    for(int i=0; i<K; i++)
        cout << sorted[i] << " ";
    cout << endl;


    bool notDone=true;
    int o=0, s=0;
    while(notDone)
    {
        if(o==K)
        {
            o=0;
            passes++;
        }

        if(original[o]==sorted[s])
        {
            original[o]=0;
//            sorted[s]=0;
            o++;
            s++;
        }
        else
            o++;

        /*
        if(o==K)
        {
            o=0;
            passes++;
        }
        */
        
        if(s==K)
            notDone=false;
    }



    cout << passes << endl;
    return 0;
}
