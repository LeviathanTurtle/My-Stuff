/* GOLF
 * William Wadsworth
 * Created: 11.18.2020
 * Doctored: 11.2.2023
 * CSC 1710
 * ~/csc1710/prog4/
 * 
 * 
 * [DESCRIPTION]:
 * This program analyzes a file consisting of golf data (file structure below).
 * It finds and outputs the player with the lowest number of strokes for the
 * game. The number of players and holes can be different than the number of
 * players and holes in the input file, that can be adjusted at runtime.
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ golfAnalysis.cpp -Wall -o golfAnalysis
 * 
 * To run (4 args):
 *     ./golfAnalysis <number of players> <number of holes> <input file>
 *
 *
 * [FILE STRUCTURE]:
 * <pars for hole 1> <pars for hole 2> ... <pars for hole n>
 * <player name 1>
 * <player name 2>
 * ...
 * <player name m>
 * <P1 strokes for hole1> <P2 strokes for hole1> ... <Pm strokes for hole1>
 * <P1 strokes for hole2> <P2 strokes for hole2> ... <Pm strokes for hole2>
 * ...
 * <P1 strokes for holen> <P2 strokes for holen> ... <Pm strokes for holen>
 *
 * where:
 *   n = number of holes 
 *   m = number of players 
 *
 *
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed full execution
 *
 * 1 - CLI args used incorrectly
 *
 * 2 - file unable to be opened or created
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
using namespace std;

// function prototypes
// function to load the hole numbers into a title array
void makeHoles(int*, const int&);
// function to output a player and their strokes per hole
void printLine(const int&, const int&, string*, int**);
// function calculate total player strokes 
int calculateSum(const int&, const int&, int**);

int main(int argc, char* argv[])
{
    // check CLI args
    if(argc != 4) {
        cerr << "error: CLI args invalid. Enter: ./golf <players> <holes> "
             << "<input file>.\n";
        exit(1);
    }
    
    // in order of file
    const int playerCount = argv[1];
    const int holeCount = argv[2];

    // DATA ARRAYS
    // array for pars (of holes)
    int* pars = new int [holeCount];

    // array for player names
    string* names = new string [playerCount];

    // array for player strokes per hole
    // x -> players
    // y -> holes
    int** results = new int* [playerCount];
    for(int i=0; i<playerCount; i++)
        results[i] = new int [holeCount];

    // array for hole numbers, will be used as a title
    int* holes = new int [holeCount];
    makeHoles(holes,holeCount);


    // ========================================================================
    // GATHER INPUT


    // datafile variable, open file
    ifstream golfData (arv[3]);

    // read in pars
    for (int i=0; i<holeCount; i++)
        golfData >> pars[i];

    // read in names
    for (int i=0; i<playerCount; i++)
        golfData >> names[i];

    // read in strokes
    for (int i=0; i<holeCount; i++)
        for (int j=0; j<playerCount; j++)
            golfData >> results[i][j];

    // close file, no longer needed
    golfData.close();


    // ========================================================================
    // OUTPUT HOLES

    // hole number (title)
    cout << endl << "       ";
    for (int i=0; i<holeCount; i++)
        cout << setw(3) << holes[i];
    cout << " Scores" << endl << endl;

    // print each player's number of strikes per hole 
    // void printLine(const int& playerNum, const int& numPlayers, string* players, int** data)
    for(int i=0; i<playerCount; i++)
        printLine(i,playerCount,names,results);
    cout << endl;
    
    // ========================================================================
    // CALCULATE SUMS

    // array for the total strokes of the game per player
    int* sums = new int [numPlayers];

    // int calculateSum(const int& playerNum, const int& holeCount, int** data)
    for(int i=0; i<numPlayers; i++)
        sums[i] = calculateSum(i,holeCount,data);
    
    // 78
    // 78
    // 101
    // 64
    
    // ========================================================================
    // find lowest score

    // set lowest to first total stroke array element, will iterate later to 
    // find true lowest
    int lowest = sums[0];
    // index in array of winning player
    int winningPlayerIndex = 0;

    for(int i=1; i<playerCount; i++)
        // find the lowest score, making note of index in array
        if(lowest > sums[i]) {
            lowest = sums[i];
            winningPlayerIndex = i;
        }

    cout << "The winner is " << names[i] << " with a score of " << lowest << endl;

    return 0;
}


// function to load the hole numbers into a title array
void makeHoles(int* array, const int& size)
{
    for (int i=0; i<size; i++)
        array[i] = i+1;
}

// function to output a player's strokes per hole
void printLine(const int& playerNum, const int& numPlayers, string* players, int** data)
{
    cout << left << setw(9) << players[playerNum];
    for(int i=0; i<numPlayers; i++)
        cout << setw(3) << data[i][playerNum];
}

// function to output a player and their total strokes
int calculateSum(const int& playerNum, const int& holeCount, int** data)
{
    int sum = 0;
    
    // calculate number of strokes in the game
    for(int i=0; i<holeCount; i++)
        sum += data[playerNum][i];
    
    return sum;
}
