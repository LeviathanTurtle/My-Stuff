# TEST SCORE ANALYSIS -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.12.2020
# Doctored: 11.14.2023
# Python-ized: 3.25.2024
# 
# [DESCRIPTION]:
# This program takes a data file from argv and processes a number of students' test grade (number 
# of students is also provided in argv). The program then assigns a letter grade to the student
# based on their (one) test score.
# 
# [USAGE]:
# To run:
#     python3 letterGradeAssignment.py <number of students> <data file>
# 
# [DATA FILE STRUCTURE]:
# <first name> <last name> <test score>
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
#include <fstream>
#include <iomanip>
using namespace std;
"""
import sys
from dataclasses import dataclass

# --- OBJECTS ---------------------------------------------------------------------------
"""
struct studentType
{
    string studentFName;
    string studentLName;
    int testScore;
    char grade;
};
"""
# data 'struct'
@dataclass
class studentType:
    first_name: str
    last_name: str
    test_score: float # changing int to float because I want to include .xx% scores
    grade: str

# --- FUNCTIONS -------------------------------------------------------------------------
# --- READ ARRAY --------------------------------
"""
void readArray(studentType*, const int&, char*);
void readArray(studentType* array, const int& numStudents, char* filename)
{
    ifstream file (filename);
    
    for(int i=0; i<numStudents; i++) {
        file >> array[i].studentFName;
        file >> array[i].studentLName;
        file >> array[i].testScore;
    }

    file.close();
}
"""
# function to read student data into array
def readArray(array, num_students, filename):
    # open the data file
    with open(filename, 'r') as file:
        # iterate over each student in the array
        for i in range(num_students):
            # read data
            array[i].first_name = file.readline().strip()
            array[i].last_name = file.readline().strip()
            array[i].test_score = float(file.readline().strip())

# --- ASSIGN GRADE ------------------------------
"""
void assignGrade(studentType*, const int&);
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
"""
# function to assign relevant grade to each student
def assignGrade(array, num_students):
    for i in range(0,num_students):
        if(array[i].test_score >= 90 and array[i].test_score <= 100):
            array[i].grade = 'A'
        elif(array[i].test_score >= 80 and array[i].test_score < 90):
            array[i].grade = 'B'
        elif(array[i].test_score >= 70 and array[i].test_score < 80):
            array[i].grade = 'C'
        elif(array[i].test_score >= 60 and array[i].test_score < 70):
            array[i].grade = 'D'
        else:
            array[i].grade = 'F'

# --- FIND NAME SIZE ----------------------------
"""
int findNameSize(studentType*, const int&);
int findNameSize(studentType* array, const int& numStudents)
{
    int largest = strlen(array[0].studentFName + array[0].studentLName);

    for(int i=1; i<numStudents; i++)
        if(strlen(array[i].studentFName + array[i].studentLName) > largest)
            largest = strlen(array[i].studentFName + array[i].studentLName);
    
    return largest;
}
"""
# function to return the size of the largest name string
def findLargestNameSize(array, num_students) -> int:
    largest = len(array[0].first_name + array[0].last_name)
    
    for i in range(1,num_students):
        if(len(array[i].first_name + array[i].last_name) > largest):
            largest = len(array[i].first_name + array[i].last_name)
    
    return largest
    
# --- PRINT ROWS --------------------------------
"""
void printRows(studentType*, const int&, const int&);
void printRows(studentType* array, const int& numStudents, const int& largestName)
{
    const int spacing = largestName+2;
    
    for(int i=0; i<numStudents; i++)
        cout << array[i].studentFName << " " array[i].studentLName 
             << setw(spacing) << array[i].testScore << setw(12) 
             << array[i].grade << endl;
}
"""
# function to print the test grade
def printRows(array, num_students, largest_name):
    spacing = largest_name + 2
    
    for i in range(0,num_students):
        

# --- HIGHEST SCORE -----------------------------
"""
int highestScore(studentType*, const int&);
int highestScore(studentType* array, const int& numStudents)
{
    int max = array[0].testScore;
    
    for(int i=1; i<numStudents; i++)
        if(array[i].testScore > max)
            max = array[i].testScore;

    return max;
}
"""
# function to print highest test score
def highestScore(array, num_students):
    pass

# --- STUDENT SCORE -----------------------------
"""
void studentScores(studentType*, const int&, const int);
void studentScores(studentType* array, const int& numStudents, int maxScore)
{
    cout << "Student(s) with the highest score (" << maxScore << "): " << endl;

    for(int i=0; i<numStudents; i++)
        if(array[i].testScore == maxScore)
            cout << array[i].studentFName << " " << array[i].studentLName << endl;
}
"""
# function to print names of students with highest test score
def studentScore(array, num_students, max_score):
    pass

# --- MAIN ------------------------------------------------------------------------------
# --- CHECK CLI ARGS ----------------------------
"""
int main(int argc, char* argv[])
{
    if(argc != 3) {
        cerr << "error: CLI args invalid. Enter: ./REPLACETHIS <number of "
             << "students> <data file>.\n";
        exit(1);
    }
"""
if len(sys.argv) < 3:
    print("Usage: python3 letterGradeAssignment.py <number of students> <data file>")
    sys.exit(1)

# --- SETUP VARS --------------------------------
"""
    const int numStudents = atoi(argv[1]);
    
    studentType* students = new studentType [numStudents];
"""
# get number of students to process from CL
num_students = int(sys.argv[1])
# define stuct variable, file variable, open data file
students = [studentType() for _ in range(num_students)]

# --- READ ARRAY --------------------------------


    

    // read in array
    //void readArray(studentType* array, const int& numStudents, char* filename)
    readArray(students,numStudents,argv[2]);

    // assign letter grade
    //void assignGrade(studentType* array, const int& numStudents)
    assignGrade(students,numStudents);

    // find the largest name size for spacing in output
    //int findNameSize(studentType* array, const int& numStudents)
    const int largestNameSize = findLargestNameSize(students, numStudents);

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



