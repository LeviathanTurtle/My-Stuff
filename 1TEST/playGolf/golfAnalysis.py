# GOLF -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.18.2020
# Doctored: 11.2.2023
# Python-ized: 3.30.2024
# 
# [DESCRIPTION]:
# This program analyzes a file consisting of golf data (file structure below). It finds and outputs
# the player with the lowest number of strokes for the game. The number of players and holes can be
# different than the number of players and holes in the input file, that can be adjusted at runtime.
#
# [USAGE]:
# python3 golfAnalysis.py <number of players> <number of holes> <input file>
# 
# [DATA FILE STRUCTURE]:
# <pars for hole 1> <pars for hole 2> ... <pars for hole n>
# <player name 1>
# <player name 2>
# ...
# <player name m>
# <P1 strokes for hole1> <P2 strokes for hole1> ... <Pm strokes for hole1>
# <P1 strokes for hole2> <P2 strokes for hole2> ... <Pm strokes for hole2>
# ...
# <P1 strokes for holen> <P2 strokes for holen> ... <Pm strokes for holen>
# 
# where:
#   n = number of holes 
#   m = number of players 
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed full execution
# 
# 1 - CLI args used incorrectly
# 
# 2 - file unable to be opened or created

# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;
"""
import sys

# --- FUNCTIONS -------------------------------------------------------------------------
# --- MAKE HOLES --------------------------------
"""
void makeHoles(int*, const int&);
void makeHoles(int* array, const int& size)
{
    for (int i=0; i<size; i++)
        array[i] = i+1;
}
"""
# function to load the hole numbers into a title array
def makeHoles(array, size):
    for i in range(size):
        array[i] = i+1

# --- PRINT LINE --------------------------------
"""
void printLine(const int&, const int&, string*, int**);
void printLine(const int& playerNum, const int& numPlayers, string* players, int** data)
{
    cout << left << setw(9) << players[playerNum];
    for(int i=0; i<numPlayers; i++)
        cout << setw(3) << data[i][playerNum];
}
"""
# function to output a player and their strokes per hole
def printLine(player_num, num_players, players, data):
    print(f"{players[player_num]:<9}", end="")
    
    for i in range(num_players):
        print(f"{data[i][player_num]:<3}")

# --- CALCULATE SUM -----------------------------
"""
int calculateSum(const int&, const int&, int**);
int calculateSum(const int& playerNum, const int& holeCount, int** data)
{
    int sum = 0;
    
    for(int i=0; i<holeCount; i++)
        sum += data[playerNum][i];
    
    return sum;
}
"""
# function calculate total player strokes 
def calculateSum(player_num, hole_count, data) -> int:
    sum = 0
    
    # calculate number of strokes in the game
    for i in range(hole_count):
        sum += data[player_num][i]
    
    return sum

# --- MAIN ------------------------------------------------------------------------------
# --- CHECK CLI ARGS ----------------------------
"""
int main(int argc, char* argv[])
{
    if(argc != 4) {
        cerr << "error: CLI args invalid. Enter: ./golf <players> <holes> "
             << "<input file>.\n";
        exit(1);
    }
    
    const int playerCount = atoi(argv[1]);
    const int holeCount = atoi(argv[2]);
"""
# check CLI args
if len(sys.argv) != 4:
    print("Usage: python3 golfAnalysis.py <number of players> <number of holes> <input file>")
    exit(1)

# in order of file
player_count = sys.argv[1]
hole_count = sys.argv[2]

# --- DATA ARRAYS -------------------------------
"""
    int* pars = new int [holeCount];

    string* names = new string [playerCount];

    int** results = new int* [playerCount];
    for(int i=0; i<playerCount; i++)
        results[i] = new int [holeCount];

    int* holes = new int [holeCount];
    makeHoles(holes,holeCount);
"""
# array for pars (of holes)
pars = [int() for _ in range(hole_count)]

# array for player names
names = [str() for _ in range(player_count)]

# array for player strokes per hole
# x -> players
# y -> holes
results = [[0 for _ in range(hole_count)] for _ in range(player_count)]

# array for hole numbers, will be used as a title
holes = [int() for _ in range(hole_count)]

# --- GATHER INPUT ------------------------------
"""
    ifstream golfData (argv[3]);

    for (int i=0; i<holeCount; i++)
        golfData >> pars[i];

    for (int i=0; i<playerCount; i++)
        golfData >> names[i];

    for (int i=0; i<holeCount; i++)
        for (int j=0; j<playerCount; j++)
            golfData >> results[i][j];

    golfData.close();
"""
# datafile variable, open file
with open(sys.argv[3], 'r') as golf_data:
    # read in pars
    for i in range(hole_count):
        pars[i] = int(golf_data.readline().strip())

    # read in names
    for i in range(player_count):
        names[i] = int(golf_data.readline().strip())

    # read in strokes
    for i in range(hole_count):
        for j in range(player_count):
            results[i][j] = int(golf_data.readline().strip())

# --- OUTPUT HOLES ------------------------------
"""
    cout << endl << "       ";
    for (int i=0; i<holeCount; i++)
        cout << setw(3) << holes[i];
    cout << " Scores" << endl << endl;

    for(int i=0; i<playerCount; i++)
        printLine(i,playerCount,names,results);
    cout << endl;
"""
# hole number (title)
print("\n       ", end="")
for i in range(hole_count):
    print(f"{holes[i]:>3}", end="")
print(" Scores\n")

# print each player's number of strikes per hole
for i in range(player_count):
    printLine(i,player_count,names,results)
print()

# --- CALCULATE SUMS ----------------------------
"""
    int* sums = new int [playerCount];

    for(int i=0; i<playerCount; i++)
        sums[i] = calculateSum(i,holeCount,data);
"""
# array for the total strokes of the game per player
sums = [int() for _ in range(player_count)]

for i in range(player_count):
    sums[i] = calculateSum(i,hole_count,results)

# --- FIND LOWEST SCORE -------------------------
"""
    int lowest = sums[0];
    int winningPlayerIndex = 0;

    for(int i=1; i<playerCount; i++)
        if(lowest > sums[i]) {
            lowest = sums[i];
            winningPlayerIndex = i;
        }

    cout << "The winner is " << names[i] << " with a score of " << lowest << endl;

    return 0;
}
"""
# set lowest to first total stroke array element, will iterate later to 
# find true lowest
lowest = sums[0]
# index in array of winning player
winning_player_index = 0

for i in range(1,player_count):
    # find the lowest score, making note of index in array
    if lowest > sums[i]:
        lowest = sums[i]
        winning_player_index = i

print(f"The winner is {names[i]} with a score of {lowest}")

