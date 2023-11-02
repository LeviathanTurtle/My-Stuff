/*
   William Wadsworth
   Dr. Williams
   csc2710
   3.28.22
 */

#define INFINITY 2000

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

void floyd(int, /*const int W[][100],*/ int D[][100], int P[][100]); // why the 100 in the second dimension?
void path(int W[][100], int P[][100], int, int);
void printArray(int W[][100], int);

int main()
{
    string file;
    ifstream data;
    int vertices, edges;

    cout << "enter datafile: ";
    cin >> file;
    data.open(file);

    data >> vertices >> edges; // nodes and vertices
    int W[vertices][100]; // also here?
    int P[vertices][100];

    for (int i = 0; i < vertices; i++)
        for (int j = 0; j < vertices; j++)
        {
            W[i][j] = INFINITY;
            P[i][j] = INFINITY;
        }

    string v1, v2;
    char hold;
    int temp1, temp2, templength;
    for (int i = 0; i < edges; i++)
    {
        data >> v1;
        hold = v1[1];
        temp1 = hold - '0';

        data >> v2;
        hold = v2[1];
        temp2 = hold - '0';
    
        data >> templength;
        W[temp1][temp2] = templength;
    }

    for (int i = 0; i < vertices; i++) // diagonal that Chris oh so loves
        W[i][i] = 0;

    cout << endl << "inital matrix (W): " << endl;
    printArray(W, vertices);
    cout << endl;

    floyd(vertices, W, P);
    cout << "shortest path: " << endl;
    printArray(W, vertices);
    cout << endl;

    cout << "stops matrix (P): " << endl;
    printArray(P, vertices);
    cout << endl;

    cout << "the shortest path from " << temp1 << " to " << temp2 << " is: ";
    path(W, P, temp1, temp2);

    cout << endl;
    data.close();
    return 0;
}

void floyd(int n, /* const int W[][100],*/ int D[][100], int P[][100])
{
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                if (D[i][k] + D[k][j] < D[i][j])
                {
                    P[i][k] = k;
                    D[i][j] = D[i][k] + D[k][j];
                }
            }

}

void path(int W[][100], int P[][100], int q, int r)
{
    if (P[q][r] != INFINITY)
    {
        path(W, P, q, W[q][r]);
        path(W, P, W[q][r], r);
    }
    else
    {
        if (W[q][r] == INFINITY)
            cout << "no path from " << q << " to " << r << endl;
        else
            cout << " v" << P[q][r];
    }
}

void printArray(int W[][100], int A)
{
    cout << "   ";
    for (int i = 0; i < A; i++)
        cout << setw(5) << "v" << i + 1;
    cout << endl;

    for (int i = 1; i <= A; i++)
    {
        cout << "v" << i << " ";
        for (int j = 1; j <= A; j++)
        {
            if (W[i][j] == INFINITY)
                cout << setw(6) << "-"; // INF
            else
                cout << setw(6) << W[i][j];
        }
        cout /*<< endl*/ << endl;
    }
}
