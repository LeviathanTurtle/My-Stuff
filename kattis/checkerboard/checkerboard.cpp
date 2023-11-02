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
    bool pass=true;
    int wcnt=0, bcnt=0;
    int first[dim];
    // horizontal
    for(int i=0; i<dim; i++)
    {
        first[i]=board[0][i];
        for(int j=0; j<dim; j++)
        {
            first[j]=board[i][j];
            if(board[i][j]=='W')
                wcnt++;
            else
                bcnt++;

//            if(wcnt!=bcnt)
//                pass=false;
        }
        for(int i=0; i<dim; i++)
        {
            if(first[i]=='W')
                wcnt++;
            else
                bcnt++;
        }
//        wcnt=bcnt=0;
    }
/*
    // vertical
    for(int i=0; i<dim; i++)
    {
        for(int j=0; j<dim; j++)
        {
            if(board[j][i]=='W')
                wcnt++;
            else
                bcnt++;

//            if(wcnt!=bcnt)
//                pass=false;
        }
//        wcnt=bcnt=0;
    }
*/

    //if(pass)
    if(wcnt%dim==0 && bcnt%dim==0)
        cout << "1" << endl;
    else
        cout << "0" << endl;

    return 0;
}
