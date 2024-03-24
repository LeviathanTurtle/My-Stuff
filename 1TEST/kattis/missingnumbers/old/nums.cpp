#include <iostream>
using namespace std;

int main ()
{
    // read-in number of lines, 
    // correct couting number, 
    // read-in holder, 
    // missing numbers array: hardcoded to max size to avoid dynamic allocation
    int templine, num=1, temp, missing[100];

    // read in number of lines
    cin >> templine;
    // convert number of lines to const so compiler doesn't complain
    const int lines = templine;

    // go through lines, if the correct number is missing, add 
    // correct number (num) to array of missing numbers. if read-in 
    // number (temp) is correct, increment correct number and 
    for(int i=0; i<lines; i++)
    {
        cin >> temp;
        while(num!=temp)
        {
            missing[num-1]=num;
            num++;
            //break;
        }
        num++;
        missing[num-1]=0;
    }
    // since num was incremented before loop fails, decrement to 
    // have final count number (last number counting to)
    num--;

    // go through missing numbers array, if the sum of contents=0,
    // output "good job" (they counted correctly)
    int cnt=0;
    for(int i=0; i<num; i++)
        cnt+=missing[i];
    if(cnt==0)
        cout << "good job" << endl;
    else
        for(int i=0; i<num; i++)
            if(missing[i]!=0)
                cout << "i= " << i << " " << missing[i] << endl;


    return 0;
}
