# SOCCER PLAYER STATISTICS -- V.PY
# William Wadsworth
# CSC1710
# Created: 1.12.2021
# Doctored: 11.2.2023
# Python-ized: 3.30.2024
# 
# [DESCRIPTION]:
# This program will store a soccer player's name, position, number of games played, total goals
# made, number of shots taken, and total number of minutes played. It will then compute the shot
# percentage and output the information in a formatted table.
# 
# [USAGE]:
# python3 soccerStats <input file>
# 
# [INPUT FILE STRUCTURE]:
# first_name last_name position games goals shots minutes
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed a full execution
# 
# 1 - CLI args used incorrectly
# 
# 2 - file unable to be opened or created

# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <iomanip>
#using namespace std;

#define MAX_SIZE 30
"""
import sys
from dataclasses import dataclass

MAX_SIZE = 30

# --- OBJECTS ---------------------------------------------------------------------------
"""
struct player {
    string name;
    string position;
    int games;
    int goals;
    int shots;
    int minutes;
};
"""
@dataclass
class player:
    name: str
    position: str
    games: int
    goals: int
    shots: int
    minutes: int

# --- FUNCTIONS -------------------------------------------------------------------------
# --- LOAD DATA ---------------------------------
"""
void loadData(const char*, player[], int&);
void loadData(const char* filename, player array[], int& count)
{
    ifstream inputFile (filename);

    if(!inputFile) {
        cerr << "error: file unable to be opened or created (provided name: " << filename << ").\n";
        exit(2);
    }
    
    string name, position;
    int games, goals, shots, minutes;
    inputFile >> name >> position >> games >> goals >> shots >> minutes;

    while (cin && count < MAX_SIZE) {
        array[count].name = name;
        array[count].position = position;
        array[count].games = games;
        array[count].goals = goals;
        array[count].shots = shots;
        array[count].minutes = minutes;

        count++;
        inputFile >> name >> position >> games >> goals >> shots >> minutes;
    }

    inputFile.close();
}
"""
# this function reads in data about players (name, position, total games played, goals, shots made,
# and minutes played. Store data in an array.
# Pre-condition: the player array references the array that will be loaded
#                with the player data
# Post-condition: the player array will be loaded with the data found in
#                 the data file, but not exceeding the max of 30
# Assumption: if the player's name can be read, assume that the position,
#             games played, goals, shots, and minutes played follows.
def loadData(filename, array, count) -> int:
    # open file
    with open(filename, 'r') as file:
        # check file was able to be opened
        if not file:
            sys.stderr.write(f"error: file unable to be opened or created (provided name: {filename}).")
            exit(2)
        
        # get first player + stats
        name, position, games, goals, shots, minutes = file.readline().split()
        
        # get the rest of the players + stats
        for _ in range(count, MAX_SIZE):
            array[count].name = name
            array[count].position = position
            array[count].games = int(games)
            array[count].goals = int(goals)
            array[count].shots = int(shots)
            array[count].minutes = int(minutes)
            
            data = file.readline().split()
            if not data:
                break
            name, position, games, goals, shots, minutes = data
    
    return count

# --- PRINT DATA --------------------------------
"""
void printData(player[], const int&);
void printData(player team[], const int& n)
{
    cout << fixed << showpoint << setprecision(2); 
    
    cout << setw(10) << right << "HPU Women's Soccer Stats" << endl;
    cout << setw(11) << left << "Name" << setw(12) << left << "Position";
    cout << setw(4) << right << "GP" << setw(4) << right << "G";
    cout << setw(6) << right << "SH" << setw(7) << right << "Mins";
    cout << setw(8) << right << "Shot%" << endl << endl;
    
    for (int i=0; i<n; i++) {
        cout << left << setw(11) << team[i].name;
        cout << left << setw(12) << team[i].position;
        cout << right << setw(4) << team[i].games;
        cout << right << setw(4) << team[i].goals;
        cout << right << setw(6) << team[i].shots;
        cout << right << setw(7) << team[i].minutes;

        if(team[i].shots == 0)
            cout << right << setw(8) << "0.0%" << endl;
        else
            cout << right << setw(7) << team[i].goals*100 / team[i].shots; << "%" << endl;
    }
}
"""
# this function output player data stored in the array (name, position, total games played, goals,
# shots made, and minutes played)
# Pre-condition: the player array (team[]) is loaded with player data for n
#                players
# Post-condition: the player array will be printed, no changes made
def printData(team, n):
    # titles
    print(f"{'HPU Soccer Stats':>10}")
    print(f"{'Name':<11}{'Position':<12}{'GP':>4}{'G':>4}{'SH':>6}{'Mins':>7}{'Shot %':>8}")
    
    # individual data
    for i in range(0,n):
        print(f"{team[i].name:<11}{team[i].position:<12}{team[i].games:>4}{team[i].goals:>4}{team[i].shots:>6}{team[i].minutes:>7}", end="")

        if(team[i].shots == 0): # this is to avoid dividing by 0
            print(f"{'0.0%':>8}")
        else:
            print(f"{team[i].goals * 100 / team[i].shots:>7.2f}%")

# --- MAIN ------------------------------------------------------------------------------
# --- CHECK CLI ARGS ----------------------------
"""
int main (int argc, char* argv[])
{
    if(argc != 2) {
        cerr << "error: CLI args used incorrectly. Proper execution: ./exe <input file>.\n";
        exit(1);
    }
"""
# check that CLI args are used correctly
if len(sys.argv) != 2:
    sys.stderr.write("Usage: python3 soccerStats.py <input file>")
    exit(1)

# --- LOAD AND PRINT ----------------------------
"""
    player db[MAX_SIZE];
    int count = 0;
    
    loadData(argv[1],db,count);
    printData(db,count);
    
    cout << endl << "Player count: " << count << endl;
    
    return 0;
}
"""
# database array
db = [player() for _ in range(MAX_SIZE)]

# load array, keeping array size
count = loadData(sys.argv[1],db,0)
printData(db,count)

print(f"Player count: {count}")

