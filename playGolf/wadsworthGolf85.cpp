/*
   William Wadsworth
   CSC1710
   11.18.20
   ~/csc1710/prog4/
   Let's play golf
   P.S. I'm not sure which percentage this is worth, but it is at least a C level without the strokes above or below par. 
      The output is neatly aligned with the hole number and total score. I have 3 functions
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;

void makeHoles(int array[], ifstream& data);
void printLine(string name, string user[], int data[][4], int pars[]);
int calculateSum(string name, int data[][4]);

int main()
{
   // in order of file
   int pars[18];
   string names[4];
   int results[18][4];

   int holes[18];

   // data variable, open file
   ifstream golfData;
   golfData.open("golfData.txt");

   // make holes array
   makeHoles(holes, golfData);

   // read in pars
   for (int i = 0; i < 18; i++)
      golfData >> pars[i];

   // read in names
   for (int i = 0; i < 4; i++)
      golfData >> names[i];

   // READ IN STROKES
   for (int r = 0; r < 18; r++)
      for (int c = 0; c < 4; c++)
         golfData >> results[r][c];
   cout << endl;

   // ===================================================================================
   // OUTPUT HOLES
   cout << "       ";
   for (int i = 0; i < 18; i++)
      cout << setw(3) << holes[i];
   cout << " Scores" << endl << endl;

   // WILLIAM RESULT
   string name1 = names[0];
   printLine(name1, names, results, pars);

   // JEFFERY RESULT
   string name2 = names[1];
   printLine(name2, names, results, pars);

   // WILL RESULT
   string name3 = names[2];
   printLine(name3, names, results, pars);

   // KRISTY RESULT
   string name4 = names[3];
   printLine(name4, names, results, pars);

   cout << endl;
    
   // ===================================================================================
   // calculate sums, store in array
   int sums[4];
    
   // 78
   sums[0] = calculateSum(name1, results);
    
   // 78
   sums[1] = calculateSum(name2, results);

   // 101
   sums[2] = calculateSum(name3, results);

   // 64
   sums[3] = calculateSum(name4, results);
    
   // ===================================================================================
   // find lowest score
   int lowest = sums[0];
    
   if (lowest > sums[1])
   {
      lowest = sums[1];
      cout << "The winner is " << name2 << " with a score of " << lowest << endl;
   }
   else if (lowest > sums[2])
   {
      lowest = sums[2];
      cout << "The winner is " << name3 << " with a score of " << lowest << endl;
   }
   else if (lowest > sums[3])
   {
      lowest = sums[3];
      cout << "The winner is " << name4 << " with a score of " << lowest << endl;
   }
   else
      cout << "The winner is " << name1 << " with a score of " << lowest << endl;

   golfData.close();
   return 0;
}

void makeHoles(int array[], ifstream& data)
{
   for (int i = 0; i < 18; i++)
      array[i] = i + 1;
}

void printLine(string name, string user[], int data[][4], int pars[])
{
   if (name == "William")
   {
      int WMsum = 0;
      cout << left << setw(9) << user[0];
      for (int i = 0; i < 18; i++)
         cout << setw(3) << data[i][0];
      for (int i = 0; i < 18; i++)
         WMsum += data[i][0];
      cout << WMsum;
   }
   else if (name == "Jeffery")
   {
      int JYsum = 0;
      cout << left << setw(9) << user[1];
      for (int i = 0; i < 18; i++)
         cout << setw(3) << data[i][1];
      for (int i = 0; i < 18; i++)
         JYsum += data[i][1];
      cout << JYsum;
   }
   else if (name == "Will")
   {
      int WLsum = 0;
      cout << left << setw(9) << user[2];
      for (int i = 0; i < 18; i++)
         cout << setw(3) << data[i][2];
      for (int i = 0; i < 18; i++)
         WLsum += data[i][2];
      cout << WLsum;
   }
   else
   {
      int KYsum = 0;
      cout << left << setw(9) << user[3];
      for (int i = 0; i < 18; i++)
         cout << setw(3) << data[i][3];
      for (int i = 0; i < 18; i++)
         KYsum += data[i][3];
      cout << KYsum;
   }
   cout << endl;
}

int calculateSum(string name, int data[][4])
{
   if (name == "William")
   {
      int WMsum = 0;
      for (int i = 0; i < 18; i++)
         WMsum += data[i][0];
      return WMsum;
   }
   else if (name == "Jeffery")
   {
      int JYsum = 0;
      for (int i = 0; i < 18; i++)
         JYsum += data[i][1];
      return JYsum;
   }
   else if (name == "Will")
   {
      int WLsum = 0;
      for (int i = 0; i < 18; i++)
         WLsum += data[i][2];
      return WLsum;
   }
   else
   {
      int KYsum = 0;
      for (int i = 0; i < 18; i++)
         KYsum += data[i][3];
      return KYsum;
   }
}
