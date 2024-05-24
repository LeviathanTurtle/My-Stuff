/* TEST SCORE ANALYSIS
 * William Wadsworth
 * Created: 11.12.2020
 * Doctored: 11.14.2023
 * CSC1710
 * ~/csc1710/lab12
 * 
 * 
 * [DESCRIPTION]:
 * This program takes a data file from argv and processes a number of students' test grade (number
 * of students is also provided in argv). The program then assigns a letter grade to the student
 * based on their (one) test score. The binary was last compiled on 5.24.2024.
 *
 *
 * [COMPILE/RUN]:
 * To compile:
 *     g++ letterGradeAssignment.cpp -Wall -o letterGradeAssignment
 *
 * To run (3 args):
 *     ./letterGradeAssignment <number of students> <data file>
 *
 *
 * [DATA FILE STRUCTURE]:
 * <first name> <last name> <test score>
 *
 *
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 *
 * 1 - CLI args used incorrectly
 *
 * 2 - file unable to be opened or created
*/


// libraries
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;


// data struct
struct studentType
{
    string studentFName;
    string studentLName;
    int testScore;
    char grade;
};


// FUNCTION PROTOTYPES
// function to read student data into array
void readArray(studentType*, const int&, char*);

// function to assign relevant grade to each student
void assignGrade(studentType*, const int&);

// function to return the size of the largest name string
int findNameSize(studentType*, const int&);

// function to print the test grade
void printRows(studentType*, const int&, const int&);

// function to print highest test score
int highestScore(studentType*, const int&);

// function to print names of students with highest test score
void studentScores(studentType*, const int&, const int);


int main(int argc, char* argv[])
{
    // process CL
    if(argc != 3) {
        cerr << "error: CLI args invalid. Enter: ./REPLACETHIS <number of "
             << "students> <data file>.\n";
        exit(1);
    }

    // get number of students to process from CL
    const int numStudents = atoi(argv[1]);
    
    // define stuct variable, file variable, open data file
    studentType* students = new studentType [numStudents];

    // read in array
    //void readArray(studentType* array, const int& numStudents, char* filename)
    readArray(students,numStudents,argv[2]);

    // assign letter grade
    //void assignGrade(studentType* array, const int& numStudents)
    assignGrade(students,numStudents);

    // find the largest name size for spacing in output
    //int findNameSize(studentType* array, const int& numStudents)
    const int largestNameSize = findNameSize(students, numStudents);

    // categories
    cout << "Student Name" << setw(20) << "Test Score" << setw(10) << "Grade" << endl << endl;
    //void printRows(studentType* array, const int& numStudents, const int& largestName)
    printRows(students,numStudents,largestNameSize);

    // show highest score
    //cout << endl << "Highest test score: " << highestScore(students,numStudents) << endl;

    // who has highest score
    //void studentScores(studentType*, const int&, const int)
    studentScores(students,numStudents,highestScore(students,numStudents));

    return 0;
}


// function to read student data into array
void readArray(studentType* array, const int& numStudents, char* filename)
{
    // create data file object
    ifstream file (filename);
    
    for(int i=0; i<numStudents; i++) {
        file >> array[i].studentFName;
        file >> array[i].studentLName;
        file >> array[i].testScore;
    }

    // close data file
    file.close();
}


// function to assign relevant grade to each student
void assignGrade(studentType* array, const int& numStudents)
{
    for(int i=0; i<numStudents; i++) {
        if(array[i].testScore >= 90 && array[i].testScore <= 100)
            array[i].grade = 'A';
        else if(array[i].testScore >= 80 && array[i].testScore < 90)
            array[i].grade = 'B';
        else if(array[i].testScore >= 70 && array[i].testScore < 80)
            array[i].grade = 'C';
        else if(array[i].testScore >= 60 && array[i].testScore < 70)
            array[i].grade = 'D';
        else
            array[i].grade = 'F';
    }
}


// function to return the size of the largest name string
int findNameSize(studentType* array, const int& numStudents)
{
    //int largest = strlen(array[0].studentFName) + strlen(array[0].studentLName);
    size_t largest = array[0].studentFName.length() + array[0].studentLName.length();

    for(int i=1; i<numStudents; i++)
        //if(strlen(array[0].studentFName)+strlen(array[0].studentLName) > largest)
        if(array[i].studentFName.length() + array[i].studentLName.length() > largest)
            //largest = strlen(array[0].studentFName) + strlen(array[0].studentLName);
            largest = array[i].studentFName.length() + array[i].studentLName.length();
    
    return largest;
}


// function to print the test grade
void printRows(studentType* array, const int& numStudents, const int& largestName)
{
    const int spacing = largestName+2;
    
    for(int i=0; i<numStudents; i++)
        cout << array[i].studentFName << " " << array[i].studentLName 
             << setw(spacing) << array[i].testScore << setw(12) 
             << array[i].grade << endl;
}


// function to print the test grade
int highestScore(studentType* array, const int& numStudents)
{
    int max = array[0].testScore;
    
    for(int i=1; i<numStudents; i++)
        if(array[i].testScore > max)
            max = array[i].testScore;

    return max;
}


// function to print names of students with highest test score
void studentScores(studentType* array, const int& numStudents, int maxScore)
{
    cout << "Student(s) with the highest score (" << maxScore << "): " << endl;

    for(int i=0; i<numStudents; i++)
        if(array[i].testScore == maxScore)
            cout << array[i].studentFName << " " << array[i].studentLName << endl;
}
