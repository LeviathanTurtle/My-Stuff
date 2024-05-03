/* SOCCER PLAYER STATISTICS
 * William Wadsworth
 * Created: 1.12.2021
 * Doctored: 11.2.2023
 * ~/csc1720/lab1/wadsworthlab1.cpp
 *
 *
 * [DESCRIPTION]:
 * This program will store a soccer player's name, position, number of games 
 * played, total goals made, number of shots taken, and total number of minutes
 * played. It will then compute the shot percentage and output the information
 * in a formatted table.
 *
 *
 * [COMPILE/RUN]:
 * To compile:
 *     g++ soccerStats.cpp -Wall -o soccerStats
 *
 * To run (2 args):
 *      ./soccerStats <input file>
 * 
 * [INPUT FILE STRUCTURE]:
 * first_name last_name position games goals shots minutes
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - CLI args used incorrectly
 * 
 * 2 - file unable to be opened or created
*/

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

struct player {
    string name;
    string position;
    int games;
    int goals;
    int shots;
    int minutes;
};

#define MAX_SIZE 30

// prototypes
void loadData(const char*, player[], int&);
void printData(player[], const int&);

int main (int argc, char* argv[])
{
    // check that CLI args are used correctly
    if(argc != 2) {
        cerr << "error: CLI args used incorrectly. Proper execution: ./exe <input file>.\n";
        exit(1);
    }
    
    // database array
    player db[MAX_SIZE];
    // array index counter
    int count = 0;
    
    loadData(argv[1],db,count);
    printData(db,count);
    
    cout << endl << "Player count: " << count << endl;
    
    return 0;
}

// this function reads in data about players (name, position, total games
// played, goals, shots made, and minutes played. Store data in an array.

/* Pre-condition: the player array references the array that will be loaded
 *                with the player data
 * Post-condition: the player array will be loaded with the data found in
 *                 the data file, but not exceeding the max of 30
 * Assumption: if the player's name can be read, assume that the position,
 *             games played, goals, shots, and minutes played follows.
*/
void loadData(const char* filename, player array[], int& count)
{
    // open file
    ifstream inputFile (filename);
    // check file was able to be opened
    if(!inputFile) {
        cerr << "error: file unable to be opened or created (provided name: " << filename << ").\n";
        exit(2);
    }
    
    // get first player + stats
    string name, position;
    int games, goals, shots, minutes;
    inputFile >> name >> position >> games >> goals >> shots >> minutes;

    // get the rest of the players + stats
    while (cin && count < MAX_SIZE) {
        array[count].name = name;
        array[count].position = position;
        array[count].games = games;
        array[count].goals = goals;
        array[count].shots = shots;
        array[count].minutes = minutes;
        // increment index counter for next player
        count++;
        inputFile >> name >> position >> games >> goals >> shots >> minutes;
    }

    inputFile.close();
}

// this function output player data stored in the array (name, position, total
// games played, goals, shots made, and minutes played)

/* Pre-condition: the player array (team[]) is loaded with player data for n
 *                players
 * Post-condition: the player array will be printed, no changes made
 */
void printData(player team[], const int& n)
{
    // set up float output
    cout << fixed << showpoint << setprecision(2); 
    
    // titles
    cout << setw(10) << right << "HPU Women's Soccer Stats" << endl;
    cout << setw(11) << left << "Name" << setw(12) << left << "Position";
    cout << setw(4) << right << "GP" << setw(4) << right << "G";
    cout << setw(6) << right << "SH" << setw(7) << right << "Mins";
    cout << setw(8) << right << "Shot%" << endl << endl;
    
    // individual data
    for (int i=0; i<n; i++) {
        cout << left << setw(11) << team[i].name;
        cout << left << setw(12) << team[i].position;
        cout << right << setw(4) << team[i].games;
        cout << right << setw(4) << team[i].goals;
        cout << right << setw(6) << team[i].shots;
        cout << right << setw(7) << team[i].minutes;

        if(team[i].shots == 0) // this is to avoid dividing by 0
            cout << right << setw(8) << "0.0%" << endl;
        else
            cout << right << setw(7) << team[i].goals*100 / team[i].shots << "%" << endl;
    }
}

