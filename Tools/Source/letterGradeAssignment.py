# TEST SCORE ANALYSIS -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.12.2020
# Doctored: 11.14.2023
# 
# Python-ized: 3.25.2024
# Updated 8.17.2024: PEP 8 Compliance
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
from sys import argv, stderr, exit
from dataclasses import dataclass
from typing import List

DEBUG: bool = False

# --- OBJECTS ---------------------------------------------------------------------------
# data 'struct'
@dataclass
class Student:
    first_name: str = ""
    last_name: str = ""
    test_score: float = 0.0 # float because I want to include .xx% scores
    grade: str = ""


# --- FUNCTIONS -------------------------------------------------------------------------
# --- READ ARRAY --------------------------------
# pre-condition: students is a list of Student objects, num_students is the number of students to
#                read, filename is the name of the file containing student data
# post-condition: each student in the array is populated with first name, last name, and test score
def read_array(students: List[Student], num_students: int, filename: str) -> None:
    """Assign a letter grade to each student based on their test score."""
    
    if DEBUG:
        print("Entering read_array...")
        
    # open the data file
    with open(filename, 'r') as file:
        # iterate over each student in the array
        for i in range(num_students):
            # read data
            students[i].first_name = file.readline().strip()
            students[i].last_name = file.readline().strip()
            students[i].test_score = float(file.readline().strip())
    
    if DEBUG:
        print("Exiting read_array.")


# --- ASSIGN GRADE ------------------------------
# pre-condition: students is a list of Student objects, num_students is the number of students in
#                the array
# post-condition: each student in the array is assigned a letter grade based on their test score
def assign_grade(students: List[Student], num_students: int) -> None:
    """Assign a letter grade to each student based on their test score."""
    
    if DEBUG:
        print("Entering assign_grade...")
        
    for i in range(num_students):
        if 90 <= students[i].test_score <= 100:
            students[i].grade = 'A'
        elif 80 <= students[i].test_score < 90:
            students[i].grade = 'B'
        elif 70 <= students[i].test_score < 80:
            students[i].grade = 'C'
        elif 60 <= students[i].test_score < 70:
            students[i].grade = 'D'
        else:
            students[i].grade = 'F'
    
    if DEBUG:
        print("Exiting assign_grade.")

# --- FIND NAME SIZE ----------------------------
# pre-condition: students is a list of Student objects, num_students is the number of students in
#                the array
# post-condition: returns the length of the longest full name (first name + last name)
def find_largest_name_size(students: List[Student], num_students: int) -> int:
    """Find the size of the largest combined first and last name in the array."""
    
    if DEBUG:
        print("Entering find_largest_name_size...")
        
    largest = len(students[0].first_name + students[0].last_name)
    
    for i in range(1,num_students):
        full_name_length = len(students[i].first_name + students[i].last_name)
        if(full_name_length > largest):
            largest = full_name_length
    
    if DEBUG:
        print("Exiting find_largest_name_size.")
    return largest
    

# --- PRINT ROWS --------------------------------
# pre-condition: students is a list of Student objects, num_students is the number of students in
#                the array, largest_name_size is the length of the longest full name in the array
# post-condition: outputs the names, test scores, and grades of all students
def print_rows(students: List[Student], num_students: int, largest_name_size: int) -> None:
    """Print the list of students with their test scores and grades."""
    
    if DEBUG:
        print("Entering print_rows...")
        
    spacing = largest_name_size + 2
    
    for i in range(0,num_students):
        print(f"{students[i].first_name} {students[i].last_name:<{spacing}} {students[i].test_score:<12} {students[i].grade}")
    
    if DEBUG:
        print("Exiting print_rows.")


# --- HIGHEST SCORE -----------------------------
# pre-condition: students is a list of Student objects, num_students is the number of students in
#                the array
# post-condition: returns the highest test score found in the array
def highest_score(students: List[Student], num_students: int) -> float:
    """Determine the highest test score among all students."""
    
    if DEBUG:
        print("Entering highest_score...")
        
    max_score = students[0].test_score
    
    for i in range(1,num_students):
        if(students[i].test_score > max_score):
            max_score = students[i].test_score
    
    if DEBUG:
        print("Exiting highest_score.")
    return max_score


# --- STUDENT SCORE -----------------------------
# pre-condition: students is a list of Student objects, num_students is the number of students in
#                the array, max_score is the highest test score in the array
# post-condition: outputs the names of students who achieved the highest score
def student_scores(students: List[Student], num_students: int, max_score: float) -> None:
    """Print the names of the students who achieved the highest test score."""
    
    if DEBUG:
        print("Entering highest_score...")
        
    print(f"Student(s) with the highest score ({max_score}): ")
    
    for i in range(num_students):
        if(students[i].test_score == max_score):
            print(f"{students[i].first_name} {students[i].last_name}")
    
    if DEBUG:
        print("Exiting highest_score.")



def main():
    # --- CHECK CLI ARGS ------------------------
    if len(argv) < 3:
        stderr.write("Usage: python3 letterGradeAssignment.py <number of students> <data file>")
        exit(1)

    # --- SETUP VARS ----------------------------
    # get number of students to process from CL
    num_students = int(argv[1])
    # define stuct variable, file variable, open data file
    students: List[Student] = [Student() for _ in range(num_students)]

    # --- READ ARRAY ----------------------------
    # read in array
    read_array(students,num_students,argv[2])

    # --- ASSIGN LETTER GRADE -------------------
    # assign letter grade
    assign_grade(students,num_students)

    # --- FIND LARGEST NAME ---------------------
    # find the largest name size for spacing in output
    largest_name_size = find_largest_name_size(students,num_students)

    # --- CATEGORIES ----------------------------
    print(f"{'Student Name':<{largest_name_size+2}} {'Test Score':<12} {'Grade'}")
    print_rows(students,num_students,largest_name_size)

    # --- HIGHEST SCORE -------------------------
    # show highest score
    student_scores(students,num_students,highest_score(students,num_students))


if __name__ == "__main__":
    main()