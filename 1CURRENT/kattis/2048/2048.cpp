#include <iostream>
using namespace std;

int main()
{
    int direction, temp, board[4][4];
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
            cin >> board[i][j];
        cout << endl;
    }
    cin >> direction;
    
    if (direction == 0) // left
        for(int i=0; i<4; i++)
            for(int j=0; j<3; j++)
            {
                if(board[i][j]==board[i][j+1])
                {
                    board[i][j] += board[i][j+1];
                    board[i][j+1] = 0;
                }
                if (board[i][j]==board[i][j+2])
                {
                    board[i][j] += board[i][j+2];
                    board[i][j+2] = 0;
                }
                if (board[i][j]==board[i][j+3])
                {
                    board[i][j] += board[i][j+3];
                    board[i][j+3] = 0;
                }
            }
//    else if (direction == 1) // up
    
//    else if (direction == 2) // right
    
//    else //if (direction == 3) // down
        
    
    
    
    for(int i=0; i<4; i++)
    {
        for(int j=0; j<4; j++)
            cout << board[i][j] << " ";
        cout << endl;
    }
    
    return 0;
}
