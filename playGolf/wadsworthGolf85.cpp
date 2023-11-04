/* GOLF
 * William Wadsworth
 * Created: 11.18.2020
 * Doctored: 11.2.2023
 * CSC 1710
 * ~/csc1710/prog4/
 * 
 * 
 * [DESCRIPTION]:
 * This program
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     
 * 
 * To run (4 args):
 *     ./
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
void makeHoles(int*, const int&);
void printLine(const int&, const int&, string*, int**);

int calculateSum(string name, int data[][4]);

int main(int argc, char* argv[])
{
    // check CLI args
    if(argc != 4) {
        cerr << "error: CLI args invalid. Enter: ./golf <players> <holes> "
             << "<input file>.\n";
        exit(1);
    }
    
    // in order of file
    int playerCount = argv[1];
    int holeCount = argv[2];

    int* pars = new int [holeCount];
    string* names = new string [playerCount];
    // x -> players
    // y -> holes
    int** results = new int* [playerCount];
    for(int i=0; i<playerCount; i++)
        results[i] = new int [holeCount];

    int* holes = new int [holeCount];


    // data variable, open file
    ifstream golfData (arv[3]);


    // make holes array
    makeHoles(holes,holeCount);


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

    cout << endl << "       ";
    for (int i=0; i<holeCount; i++)
        cout << setw(3) << holes[i];
    cout << " Scores" << endl << endl;

    // print each player's number of strikes per hole 
    // void printLine(const int& playerNum, const int& numPlayers, string* players, int** data)
    for(int i=0; i<holeCount; i++)
        printLine(playerCount,names,results);
    cout << endl;
    
    // ========================================================================
    // CALCULATE SUMS

    int* sums = new int [numPlayers];

    // int calculateSum(const int& playerNum, const int& holeCount, int** data)
    for(int i=0; i<numPlayers; i++)
        sums[i] = calculateSum(i,playerCount,data);
    
    // 78
    // 78
    // 101
    // 64
    
    // ========================================================================
    // find lowest score
    int lowest = sums[0];
    
    if (lowest > sums[1]) {
        lowest = sums[1];
        cout << "The winner is " << name2 << " with a score of " << lowest << endl;
     }
     else if (lowest > sums[2]) {
        lowest = sums[2];
        cout << "The winner is " << name3 << " with a score of " << lowest << endl;
    }
    else if (lowest > sums[3]) {
        lowest = sums[3];
        cout << "The winner is " << name4 << " with a score of " << lowest << endl;
    }
    else
        cout << "The winner is " << name1 << " with a score of " << lowest << endl;

    return 0;
}


void makeHoles(int* array, const int& size)
{
    for (int i=0; i<size; i++)
        array[i] = i+1;
}


void printLine(const int& playerNum, const int& numPlayers, string* players, int** data)
{
    cout << left << setw(9) << players[playerNum];
    for(int i=0; i<numPlayers; i++) {
        cout << setw(3) << data[i][playerNum];
    }
}


int calculateSum(const int& playerNum, const int& holeCount, int** data)
{
    int sum = 0;
    
    // calculate number of strokes in the game
    for(int i=0; i<holeCount; i++)
        sum += data[playerNum][i];
    
    return sum;
}
