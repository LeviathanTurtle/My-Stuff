#include <iostream>
#include <iomanip>
using namespace std;

int main()
{
    int cardcnt, card, cnt=0;
    double sum=0;

    // read number of cards
    cin >> cardcnt;
    
    // set output to show 9 decimals
    cout << fixed << showpoint << setprecision(9);

    // if card is not negative, add to sum and increment card count
    for(int i=0; i<cardcnt; i++)
    {
        cin >> card;
        if(card>0)
        {
            sum+=card;
            cnt++;
        }
    }

    // output avg: counted card sum / counted cards
    if(cnt==0)
        cout << "0.000000000" << endl;
    else
        cout << sum/cnt << endl;

    return 0;
}
