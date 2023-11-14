/* 
 * William Wadsworth
 * Created: 
 * Doctored: 11.14.2023
 * CSC 17
 * ~/
 * 
 * 
 * [DESCRIPTION]:
 * This program 
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ 
 * 
 * To run:
 *     ./
 * 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - 
*/

//#include <stdio.h>
#include <iostream>
using namespace std;

struct sportsType
{
    string sportName;
    string teamName;
    int numOfPlayers;
    double teampayroll;
    double coachSalary;
};

// load struct
void getInfo(sportsType& athlete);
// compute average salary
double calculateSalary(sportsType athlete);
// output
void printInfo(sportsType athlete);

int main ()
{
    sportsType person;
    
    getInfo(person);
    cout << endl << "Your total salary is " << calculateSalary(person) << endl;
    printInfo(person);
    
    return 0;
}

void getInfo(sportsType& athlete)
{
    cout << "What is your sport: ";
    cin >> athlete.sportName;
    
    cout << "What is your team name: ";
    cin >> athlete.teamName;
    
    cout << "How many people are on your team: ";
    cin >> athlete.numOfPlayers;
    
    cout << "How big are your pockets: ";
    cin >> athlete.teampayroll;
    
    cout << "How much does your coach make: ";
    cin >> athlete.coachSalary;
}

double calculateSalary(sportsType athlete)
{
    return (athlete.teampayroll + athlete.coachSalary);
}

void printInfo(sportsType athlete)
{
    cout << endl << endl << endl;
    cout << "Sport: " << athlete.sportName << endl;
    cout << "Team name: " << athlete.teamName << endl;
    cout << "Number of players: " << athlete.numOfPlayers << endl;
    cout << "Team net worth: $" << athlete.teampayroll << endl;
    cout << "Coach net worth: $" << athlete.coachSalary << endl;
    cout << "Average earned: $" << calculateSalary(athlete) << endl;
}

