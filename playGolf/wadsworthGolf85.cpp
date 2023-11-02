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

void printLine(string name, string user[], int data[][4], int pars[]);
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
    int numPlayers = argv[1];
    int numHoles = argv[2];

    int* pars = new int [numHoles];
    string* names = new string [numPlayers];
    // x -> players
    // y -> holes
    int** results = new int* [numPlayers];
    for(int i=0; i<numPlayers; i++)
        results[i] = new int [numHoles];

    int* holes = new int [numHoles];


    // data variable, open file
    ifstream golfData (arv[3]);


    // make holes array
    makeHoles(holes,numHoles);


    // read in pars
    for (int i=0; i<numHoles; i++)
        golfData >> pars[i];

    // read in names
    for (int i=0; i<numPlayers; i++)
        golfData >> names[i];

    // read in strokes
    for (int i=0; i<numHoles; i++)
        for (int j=0; j<numPlayers; j++)
            golfData >> results[i][j];

    // close file, no longer needed
    golfData.close();

    // ========================================================================
    // OUTPUT HOLES

    cout << endl << "       ";
    for (int i=0; i<numHoles; i++)
        cout << setw(3) << holes[i];
    cout << " Scores" << endl << endl;


    for(int i=0; i<numPlayers; i++) {
        string name = names[i];
        printLine(name,names,results,pars);
    }


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
    
    // ========================================================================
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

void printLine(string name, string user[], int data[][4], int pars[])
{
    if (name == "William") {
        int WMsum = 0;
        cout << left << setw(9) << user[0];
        for (int i=0; i<18; i++)
            cout << setw(3) << data[i][0];
        for (int i=0; i<18; i++)
            WMsum += data[i][0];
        cout << WMsum;
    }
    else if (name == "Jeffery") {
        int JYsum = 0;
        cout << left << setw(9) << user[1];
        for (int i=0; i<18; i++)
            cout << setw(3) << data[i][1];
        for (int i=0; i<18; i++)
            JYsum += data[i][1];
        cout << JYsum;
    }
    else if (name == "Will") {
        int WLsum = 0;
        cout << left << setw(9) << user[2];
        for (int i=0; i<18; i++)
            cout << setw(3) << data[i][2];
        for (int i=0; i<18; i++)
            WLsum += data[i][2];
        cout << WLsum;
    }
    // Kristy
    else {
        int KYsum = 0;
        cout << left << setw(9) << user[3];
        for (int i=0; i<18; i++)
            cout << setw(3) << data[i][3];
        for (int i=0; i<18; i++)
            KYsum += data[i][3];
        cout << KYsum;
    }
    cout << endl;
}

int calculateSum(string name, int data[][4])
{
    if (name == "William") {
        int WMsum = 0;
        for (int i=0; i<18; i++)
            WMsum += data[i][0];
        return WMsum;
    }
    else if (name == "Jeffery") {
        int JYsum = 0;
        for (int i=0; i<18; i++)
            JYsum += data[i][1];
        return JYsum;
    }
    else if (name == "Will") {
        int WLsum = 0;
        for (int i=0; i<18; i++)
            WLsum += data[i][2];
        return WLsum;
    }
    else {
        int KYsum = 0;
        for (int i=0; i<18; i++)
            KYsum += data[i][3];
        return KYsum;
    }
}
