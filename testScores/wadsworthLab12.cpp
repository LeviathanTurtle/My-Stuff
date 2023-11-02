/*
 * William Wadsworth
 * 11.12.20
 * CSC1710
 * ~/csc1710/lab12
 * 
 * 
*/

// libraries
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

struct studentType
{
    string studentFName;
    string studentLName;
    int testScore;
    char grade;
};

// function to read student data into array
void readArray(studentType array[], ifstream& file);
// function to assign relevant grade to each student
void assignGrade(studentType array[]);
// function to print the test grade
void printRow(studentType array[]);
// function to print highest test score
int highestScore(studentType array[]);
// function to print names of students with highest test score
void studentScores(studentType array[], int maxScore);

int main()
{
    // define stuct variable, file variable, open data file
    studentType students[20];
    ifstream data;
    data.open("data.txt");

    // read in array, assign letter grade
    readArray(students, data);
    assignGrade(students);

    // categories
    cout << "Student Name" << setw(20) << "Test Score" << setw(10) << "Grade" << endl << endl;
    printRow(students);

    // show highest score
    cout << endl << "Highest test score: " << highestScore(students) << endl;

    // who has highest score
    studentScores(students, highestScore(students));

    // close file, end program
    data.close();
    return 0;
}

void readArray(studentType array[], ifstream& file)
{
    for (int i = 0; i < 20; i++) {
        file >> array[i].studentFName;
        file >> array[i].studentLName;
        file >> array[i].testScore;
    }
}

void assignGrade(studentType array[])
{
    for (int i = 0; i < 20; i++) {
        if (array[i].testScore >= 90 && array[i].testScore <= 100)
            array[i].grade = 'A';
        else if (array[i].testScore >= 80 && array[i].testScore < 90)
            array[i].grade = 'B';
        else if (array[i].testScore >= 70 && array[i].testScore < 80)
            array[i].grade = 'C';
        else if (array[i].testScore >= 60 && array[i].testScore < 70)
            array[i].grade = 'D';
        else
            array[i].grade = 'F';
    }
}

// not using a loop unfortunately because the sizes of names are different which means different spacing 
void printRow(studentType array[])
{
    int i = 0;
    
    // donald
    cout << array[i].studentFName << " " << array[i].studentLName << setw(15) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // goofy
    cout << array[i].studentFName << " " << array[i].studentLName << setw(18) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // balto
    cout << array[i].studentFName << " " << array[i].studentLName << setw(17) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // smitn
    cout << array[i].studentFName << " " << array[i].studentLName << setw(18) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // wonderful
    cout << array[i].studentFName << " " << array[i].studentLName << setw(13) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // akthar
    cout << array[i].studentFName << " " << array[i].studentLName << setw(15) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // green
    cout << array[i].studentFName << " " << array[i].studentLName << setw(17) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // egger
    cout << array[i].studentFName << " " << array[i].studentLName << setw(16) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // deer
    cout << array[i].studentFName << " " << array[i].studentLName << setw(18) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // jackson
    cout << array[i].studentFName << " " << array[i].studentLName << setw(15) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // gupta
    cout << array[i].studentFName << " " << array[i].studentLName << setw(18) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // happy
    cout << array[i].studentFName << " " << array[i].studentLName << setw(16) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // arora
    cout << array[i].studentFName << " " << array[i].studentLName << setw(17) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // june
    cout << array[i].studentFName << " " << array[i].studentLName << setw(17) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // cheng
    cout << array[i].studentFName << " " << array[i].studentLName << setw(19) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // malik
    cout << array[i].studentFName << " " << array[i].studentLName << setw(16) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // tomek
    cout << array[i].studentFName << " " << array[i].studentLName << setw(15) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // clodfelter
    cout << array[i].studentFName << " " << array[i].studentLName << setw(11) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // nields
    cout << array[i].studentFName << " " << array[i].studentLName << setw(14) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
    // norman
    cout << array[i].studentFName << " " << array[i].studentLName << setw(16) << array[i].testScore << setw(12) << array[i].grade << endl;
    i++;
}

int highestScore(studentType array[])
{
    int max = array[0].testScore;
    
    for (int i = 1; i < 20; i++)
        if (array[i].testScore > max)
            max = array[i].testScore;

    return max;
}

void studentScores(studentType array[], int maxScore)
{
    cout << "Students with the highest score (" << maxScore << "): " << endl;

    for (int i = 0; i < 20; i++)
        if (array[i].testScore == maxScore)
            cout << array[i].studentFName << " " << array[i].studentLName << endl;
}
