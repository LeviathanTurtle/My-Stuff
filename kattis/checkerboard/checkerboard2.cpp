#include <iostream>
using namespace std;

int main()
{
    // read in board size
    int dim;
    cin >> dim;

    // construct array (board) size
    char board[dim][dim];
    string read;
    for(int i=0; i<dim; i++)
    {
        cin >> read;
        for(int j=0; j<dim; j++)
            board[i][j]=read[j];
    }

/*
    // DEBUG
    for(int i=0; i<dim; i++)
    {
        for(int j=0; j<dim; j++)
            cout << board[i][j];
        cout << endl;
    }
*/

    // TESTING
    int wcnt=0, bcnt=0;

    for(int i=0; i<dim; i++)
    {
        for(int j=0; j<dim; j++)
        {
            if(board[i][j]=='B')
                bcnt++;
            else
                wcnt++;

            if(board[i][j]


    //if(pass)
    if(wcnt%dim==0 && bcnt%dim==0)
        cout << "1" << endl;
    else
        cout << "0" << endl;

    return 0;
}
