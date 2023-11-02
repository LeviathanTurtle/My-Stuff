/*
   Author: William Wadsworth
   Date: 3.22.21
   Class: CSC1720
   Code location: ~/csc1720/prog3/wadsworthProg3.cpp

   About:
      This program reads a file containing a list of names and determines who wins prizes.

   To compile:
      g++ -Wall wadsworthProg3.cpp -o testProg

   To execute:
      ./testProg
*/

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "arrayListType.h"
#include "unorderedArrayListType.h"
#include "uniqueListType.h"
using namespace std;

void load(ifstream&, unorderedArrayListType<string>&, unorderedArrayListType<string>&);
void load(uniqueListType<string>&, unorderedArrayListType<string>&, unorderedArrayListType<string>&);
string decideWinner(const string, uniqueListType<string>&);

int main()
{
    // datafile variable, input variable, raw array
    string file;
    ifstream inFile;
    unorderedArrayListType<string> firstName, lastName;

    // picking datafile
    cout << "Select a datafile to use (include file format): ";
    cin >> file;
    cout << endl;
    // repeat if no data file is open
    while (!inFile.is_open())
    {
        if (file == "list.txt")
        {
            cout << "attempting to open file " << file << "..." << endl;
            inFile.open(file);
            cout << "file successfully opened" << endl;
        }
        else
        {
            cout << "invalid filename, re-enter: ";
            cin >> file;
        }
    }

    // load raw array
    cout << endl;
    load(inFile, firstName, lastName);
    
    // create/load unique array, print contestants
    uniqueListType<string> contestants(2*lastName.maxListSize());
    cout << endl;
    load(contestants, firstName, lastName);
    cout << endl << "Contestants: " << endl;
    contestants.print();
    
    uniqueListType<string> nonWinners = contestants;
    // decide BWG
    string prize1 = "Dinner for 2 at Blue Water Grille";
    decideWinner(prize1, nonWinners);
    
    // decide WNW
    string prize2 = "4 passes to Wet N Wild";
    decideWinner(prize2, nonWinners);

    // decide AGC
    string prize3 = "$100 Amazon Gift Card";
    decideWinner(prize3, nonWinners);

    // decide PRT
    string prize4 = "4 movie passes to Palladium Regal Theater";
    decideWinner(prize4, nonWinners);

    // decide PS5
    string prize5 = "PS5";
    decideWinner(prize5, nonWinners);

    // close data file, end program
    inFile.close();
    return 0;
}

void load(ifstream& file, unorderedArrayListType<string>& FNlist, unorderedArrayListType<string>& LNlist)
{
    string first, last;
    int i = 0;

    cout << "attempting to read in contents of file..." << endl;
    file >> first >> last;
    while (!file.eof())
    {
        for (int i = 0; i < FNlist.maxListSize() && i < LNlist.maxListSize(); i++)
        {
            FNlist.insertAt(i, first);
            LNlist.insertAt(i, last);
            file >> first >> last;
        }
    }
    cout << "datafile successfully read-in" << endl;
}

void load(uniqueListType<string>& c, unorderedArrayListType<string>& firstN, unorderedArrayListType<string>& lastN)
{
    cout << "attempting to merge first/last name arrayTypes into uniqueType..." << endl;
    for (int i = 0; i < firstN.maxListSize() && i < lastN.maxListSize(); i++)
    {
        if (c.getAt(i) != firstN.getAt(i))
            c.insertAt(i, firstN.getAt(i));
        if (c.getAt(i+1) != lastN.getAt(i))
            c.insertAt(i+1, lastN.getAt(i));
    }
    cout << "first/last name arrayType merging successful" << endl;
}

string decideWinner(const string prize, uniqueListType<string>& group)
{
    ostringstream sout;
    switch (1 + (rand() % group.maxListSize()))
    {
        case 1:
        {
            sout << group.getAt(0) << " " << group.getAt(1) << " won " << prize << "!" << endl;
            group.removeAt(0);
            group.removeAt(1);
        }
        case 2:
        {
            sout << group.getAt(2) << " " << group.getAt(3) << " won " << prize << "!" << endl;
            group.removeAt(2);
            group.removeAt(3);
        }
        case 3:
        {
            sout << group.getAt(4) << " " << group.getAt(5) << " won " << prize << "!" << endl;
            group.removeAt(4);
            group.removeAt(5);
        }
        case 4:
        {
            sout << group.getAt(6) << " " << group.getAt(7) << " won " << prize << "!" << endl;
            group.removeAt(6);
            group.removeAt(7);
        }
        case 5:
        {
            sout << group.getAt(8) << " " << group.getAt(9) << " won " << prize << "!" << endl;
            group.removeAt(8);
            group.removeAt(9);
        }
        case 6:
        {
            sout << group.getAt(10) << " " << group.getAt(11) << " won " << prize << "!" << endl;
            group.removeAt(10);
            group.removeAt(11);
        }
        case 7:
        {
            sout << group.getAt(12) << " " << group.getAt(13) << " won " << prize << "!" << endl;
            group.removeAt(12);
            group.removeAt(13);
        }
        case 8:
        {
            sout << group.getAt(14) << " " << group.getAt(15) << " won " << prize << "!" << endl;
            group.removeAt(14);
            group.removeAt(15);
        }
        case 9:
        {
            sout << group.getAt(16) << " " << group.getAt(17) << " won " << prize << "!" << endl;
            group.removeAt(16);
            group.removeAt(17);
        }
        case 10:
        {
            sout << group.getAt(18) << " " << group.getAt(19) << " won " << prize << "!" << endl;
            group.removeAt(18);
            group.removeAt(19);
        }
    }
    return sout.str();
}

/* NOTE: I ended up wasting a few hours trying to figure out how to open the file without giving
 *       a specific name. I ended up just settling with a specific name and figured I'd take the 
 *       points off. Also the print function looks a little weird to me, maybe I did it wrong?
 *       My overloaded load function to combine first/last names into a unique list didn't work
 *       right either.
 */

/* load new list to sort out duplicates
 * count duplicates
 * add total # of names
 * use RNG to determine winner
 */