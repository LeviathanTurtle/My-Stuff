/* MONEY CALCULATOR
 * 
 * William Wadsworth
 * Created: 8.24.2020
 * Doctored: 10.12.2023
 * CSC1710-02
 * ~/csc1710/lab2/assignment.cpp
 * 
 * 
 * [SUMMARY]:
 * This program prompts the user for the number of 1, 5, 10, 20, 50, and 100 dollar bills they
 * have. It then calculates and outputs the total sum of money based on what the user input. The
 * binary was last compiled on 5.24.2024.
 * 
 * 
 * [USE]:
 * To compile: g++ moneyCalculator.cpp -Wall -o moneyCalculator
 * To run: ./moneyCalculator
*/


#include <iostream>
using namespace std;


int main()
{
    // bill variables
    int bills1, bills5, bills10, bills20, bills50, bills100;

    // prompt user
    cout << "Enter the number of $1 bills: ";
    // take input
    cin >> bills1;
    // if the user did not provide an int, repeatedly prompt until one is given
    if(typeid(bills1) != typeid(int)) {
        cout << "error: integer number not provided.";
        exit(1);
    }

    // repeat same steps for 5, 10, 20, 50, 100

    cout << "Enter the number of $5 bills: ";
    cin >> bills5;
    if(typeid(bills5) != typeid(int)) {
        cout << "error: integer number not provided.";
        exit(1);
    }

    cout << "Enter the number of $10 bills: ";
    cin >> bills10;
    if(typeid(bills10) != typeid(int)) {
        cout << "error: integer number not provided.";
        exit(1);
    }

    cout << "Enter the number of $20 bills: ";
    cin >> bills20;
    if(typeid(bills20) != typeid(int)) {
        cout << "error: integer number not provided.";
        exit(1);
    }

    cout << "Enter the number of $50 bills: ";
    cin >> bills50;
    if(typeid(bills50) != typeid(int)) {
        cout << "error: integer number not provided.";
        exit(1);
    }

    cout << "Enter the number of $100 bills: ";
    cin >> bills100;
    if(typeid(bills100) != typeid(int)) {
        cout << "error: integer number not provided.";
        exit(1);
    }

    // calculate total
    int total = bills1 + (5*bills5) + (10*bills10) + (20*bills20) + (50*bills50) + (100*bills100);

    cout << "You have $" << total << endl;
    return 0;
}

