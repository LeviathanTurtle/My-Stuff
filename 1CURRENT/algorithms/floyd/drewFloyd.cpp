/* Drew Faust, for csc2710
   This is an implementation of Floyd Warshall Algorithm

*/

using namespace std;
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#define INFINITY 2000

void printArray(int V[][100], int A);
void floyd (int V[][100], int P[][100], int A);
void findPath (int A, int B, int V[][100], int P[][100]);

int main ()
{
   string file, v1, v2;
   ifstream indata;
   int nodes, vertices, temp1, temp2, tempLength;

   cout << "Please enter the exact name of the file you would like to use: ";
   cin >> file;

   indata.open(file);

   indata >> nodes;
   int V[nodes][100]; //We give the second parameter a base value of 100 as C++
   int P[nodes][100]; //Doesn't like passing arrays with more than 1 dimension
                      //of variable size.

   for (int i=1; i<nodes+1; i++)
   {
      for (int j=1; j<nodes+1; j++)
      {
         V[i][j]=INFINITY;
         P[i][j]=INFINITY;
     }
   }

   indata >> vertices;

   for (int i=0; i<vertices; i++)
   {
      indata >> v1 >> v2 >> tempLength;
      temp1 = v1[1]-48;
      temp2 = v2[1]-48;
      V[temp1][temp2]=tempLength;
   }

   for (int i=1; i<nodes+1; i++)
   {
      V[i][i]=0;
   }

   cout << endl << "This is our initial matrix: " << endl;
   printArray(V, nodes);

   floyd(V, P, nodes);
   cout << "This is the shortest path matrix: " << endl;
   printArray (V, nodes);

   cout << "These are the stops taken to achieve that shortest path." << endl;
   cout << "Infinity represents that there is no path, it is the same node, or that the shortest path is direct." << endl;
   printArray (P, nodes);

   findPath (temp1, temp2, V, P);

   return 0;
}

void printArray(int V[][100], int A)
{

   cout << "   ";

   for (int i=1; i<A+1; i++)
   {
      cout << setw(5) << "v" << i;
   }
   cout << endl;

   for (int i=1; i<A+1; i++)
   {
      cout << "v" << i << " ";
      for (int j=1; j<A+1; j++)
      {
         if (V[i][j] == 2000)
            cout << setw(6) << "INF";
         else
            cout << setw(6) << V[i][j];
      }
      cout << endl << endl;
   }
}

void floyd (int V[][100], int P[][100], int A)
{
   for (int k=1; k<A+1; k++)
   {
       for (int i=1; i<A+1; i++)
       {
          for (int j=1; j<A+1; j++)
          {
             if (V[i][j]>(V[k][j]+V[i][k]))
             {
                V[i][j]=(V[k][j]+V[i][k]);
                P[i][j]=k;
             }
          }
       }
    }
}

void findPath(int A, int B, int V[][100], int P[][100])
{
   cout << "The shortest path from v" << A << " to v" << B << " is ";
   if (P[A][B] != 2000)
   {
      findPath (A, V[A][B], V, P);
      findPath (V[A][B], B, V, P);
   }
   else
   {
      if (V[A][B] == 2000)
      {
        cout << "There is no path from v" << A << " to v" << B << "." << endl;
      }
      else
      {
        cout << "The shortest path from v" << A << " to v" << B << " is direct." << endl;
      }
   }
}
