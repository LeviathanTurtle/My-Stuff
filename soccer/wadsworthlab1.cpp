/*
   Author: William Wadsworth
   Date: 1.12.20
   Class: CSC1720
   Code location: ~/csc1720/lab1/wadsworthlab1.cpp
 
   About: This program will store a soccer player's name, position, number of
          games played, total goals made, number of shots taken, and total number of
          minutes played. It will then compute the shot percentage and output the
          information in a formatted table.
 */

#include <iostream>
#include <iomanip>
using namespace std;

struct player
{
   string name;
   string position;
   int games;
   int goals;
   int shots;
   int minutes;
};

#define MAX 30

// prototypes
void loadData(player array[], int &cnt);
void printData(player team[], int n);

int main ()
{
   player db[MAX];
   int count = 0;
    
   loadData(db, count);
   printData(db, count);
    
   cout << endl << "Player count: " << count << endl;
    
   return 0;
}

/*
   loadData - read in data about players (name, position, total games played,
   goals, shots made, and minutes played. Store data in an array.
   Pre-condition: the player array references the array that will be loaded
                  with the player data
   Post-condition: the player array will be loaded with the data found in
                   the data file, but not exceeding the max of 30
   Assumption: if the player's name can be read, assume that the position,
               games played, goals, shots, and minutes played follows.
 */
void loadData(player array[], int &cnt)
{
   cnt = 0;
   string name;
   string position;
   int games;
   int goals;
   int shots;
   int minutes;
   cin >> name >> position >> games >> goals >> shots >> minutes;
   while (cin && cnt < MAX)
   {
      array[cnt].name = name;
      array[cnt].position = position;
      array[cnt].games = games;
      array[cnt].goals = goals;
      array[cnt].shots = shots;
      array[cnt].minutes = minutes;
      cnt++;
      cin >> name >> position >> games >> goals >> shots >> minutes;
   }
}

/*
   printData - output player data stored in the array (name, position, total
               games played, goals, shots made, and minutes played)
   Pre-condition: the player array (team[]) is loaded with player data for n
                  players
   Post-condition: the player array will be printed, no changes made
 */
void printData(player team[], int n)
{
   cout << setw(10) << right << "HPU Women's Soccer Stats" << endl;
   cout << setw(11) << left << "Name" << setw(12) << left << "Position";
   cout << setw(4) << right << "GP" << setw(4) << right << "G";
   cout << setw(6) << right << "SH" << setw(7) << right << "Mins";
   cout << setw(8) << right << "Shot%" << endl << endl;
    
   for (int i = 0; i < n; i++)
   {
      cout << left << setw(11) << team[i].name;
      cout << left << setw(12) << team[i].position;
      cout << right << setw(4) << team[i].games;
      cout << right << setw(4) << team[i].goals;
      cout << right << setw(6) << team[i].shots;
      cout << right << setw(7) << team[i].minutes;
      if (team[i].shots == 0)
         cout << right << setw(8) << "0.0%" << endl;
      else
      {
         double percent = team[i].goals * 100 / team[i].shots;
         cout << fixed << showpoint << setprecision(1); 
         cout << right << setw(7) << percent << "%" << endl;
      }
   }
}

