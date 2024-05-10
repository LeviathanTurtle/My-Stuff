# ARRAY MEASUREMENTS OF CENTER -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.5.2020
# Doctored: 10.25.2023
# Updated: 10.25.2023 -- added average function
# Python-ized: 3.31.2024
# 
# [DESCRIPTION]:
# This program loads an array from a file and sorts the data. It then outputs the size of the data,
# median, minimum, maximum. It will then prompt a to search the data for a value, and output the
# number of occurences. The end of the data in the file is noted by a value of -1.
# 
# [USAGE]:
# python3 dataDetails.py <file name>
# 
# [EXIT/TERMINATING CODES]:
# 0 - the user specified information was incorrect or the program successfully completed a full 
#     execution.
# 
# 1 - command line arguments were used incorrectly
# 
# 2 - the file was unable to be opened or created


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define MAX_SIZE 1000
"""
import sys
from typing import List, Tuple

MAX_SIZE = 1000

# --- FUNCTIONS -------------------------------------------------------------------------
# --- LOAD ARRAY --------------------------------
"""
void loadArray(double[], const char*, const int&);
void loadArray(double nums[], const char* filename, int& size)
{
	ifstream dataFile (filename);
 
	if(!dataFile) {
		cerr << "error: file unable to be opened or created (provided name: "
		     << filename << ").\n";
		exit(2);
	}
	
	int index = 0;

    dataFile >> nums[index];

    while (nums[index] != -1) {
        size++;
		index++;
        dataFile >> nums[index];
    }

	dataFile.close();
}
"""
# this function creates an input file of a name that is passed as a parameter from main. If the
# file was opened successfully, it reads in the data while keeping note of the size of the array.
def loadArray(filename,size) -> Tuple[List[float], int]:
    # numbers array
    nums: List[float] = []
    size = 0
    
    try:
        # open user-specified file
        with open(filename,'r') as file:
            # as long as the current value is non-terminating (-1), update the size of the array
            # and get the next one
            for line in file:
                number = float(line.strip())
                if number == -1 or len(nums) == MAX_SIZE:
                    break
                
                # add the number to the list
                nums.append(number)
                # update size
                size += 1
                
    except IOError:
        sys.stderr.write(f"error: file unable to be opened or created (provided name: {filename})")
        exit(2)

    return nums, size

# --- PRINT ARRAY -------------------------------
"""
void printArray(double[], const int&);
void printArray(double numbers[], const int& size)
{
	for(int i=0; i<size; i++)
		cout << "nums[" << i << "] = " << numbers[i] << endl;
}
"""
# this function takes an array and its size as parameters, and outputs each value in said array
# along with its location.
def printArray(numbers, size):
    for i in range(size):
        print(f"nums[{i}] = {numbers[i]}")

# --- SORT ARRAY --------------------------------
"""
void sortArray(double[], const int&);
void sortArray(double numArray[], const int& size)
{
	int hold;

	for (int pass = 1; pass < size; pass++)
		for (int i = 0; i < size - pass; i++)
			if (numArray[i] > numArray[i+1]) {
				hold = numArray[i];
				numArray[i] = numArray[i+1];
				numArray[i+1] = hold;
			}
}
"""
# this function takes an array and its size as parameters and then sorts the array.
def sortArray(num_array, size):
    # temp variable for swap
    hold: float
    
    for i in range(1,size):
        for j in range(0,size-i):
            # is the current value larger than the next?
            if num_array[i] > num_array[i+1]:
                # swap numbers
                num_array[i], num_array[i+1] = num_array[i+1], num_array[i]

# --- MEDIAN ------------------------------------
"""
double median(double[], const int&);
double median(double array[], const int& size)
{
	double med;
	
	if (size % 2 != 0) {
		med = array[size/2];
		cout << "nums[" << size/2 << "] = ";
	}
	else {
		cout << "(nums[" << size/2 << "] + nums[" << (size/2)-1 << "]) / 2 = ";
		med = (array[size/2] + array[(size/2)-1]) / 2;
	}

	return med;
}
"""
# this function takes an array and its size as parameters. The function will then calculate and
# return the median value along with its location in the array. The function assumes the array is
# already sorted.
def median(array, size) -> float:
    med: float
    
    # if the size is odd, pick the middle value
    if size%2 != 0:
        med = array[size/2]
        #print(f"nums[{size/2}] = ")
    # if the size is even, average the two middle values
    else:
        med = (array[size/2] + array[(size/2)-1]) / 2
        #print(f"(nums[{size/2}] + nums[{(size/2)-1}]) / 2 = ")
    
    return med

# --- AVERAGE -----------------------------------
"""
double average(double[], const int&);
double average(double array[], const int& size)
{
	double sum = 0;

	for(int i=0; i<size; i++)
		sum += array[i];

	return sum/size;
}
"""
# this function takes an array and its size as parameters. The function will then calculate and
# return the average of the values in the array.
def average(array, size) -> float:
    sum = 0
    
    for i in range(size):
        sum += array[i]
    
    return sum/size

# --- MINIMUM -----------------------------------
"""
double minimum(double[]);
double minimum(double array[])
{
	cout << "nums[0] = ";

	return array[0];
}
"""
# this function takes an array and its size as parameters. The function will then return the lowest
# value and its location in the array. The function assumes the array is already sorted.
def minimum(array) -> float:
    #print("nums[0] = ")
    return array[0]

# --- MAXIMUM -----------------------------------
"""
double maximum(double[], const int&);
double maximum(double array[], const int& size)
{
	cout << "nums[" << size-1 << "] = ";

	return array[size-1];
}
"""
# this function takes an array and its size as parameters. The function will then return the
# largest value and its location in the array. The function assumes the array is already sorted.
def maximum(array, size) -> float:
    #print(f"nums[{size-1}] = ")
    return array[size-1]

# --- SEARCH ARRAY ------------------------------
"""
int searchArray(double[], const int&, const int&);
int searchArray(double array[], const int& searchItem, const int& size)
{
	int matches = 0;

	for(int i=0; i<size; i++)
		if (array[i] == searchItem)
			matches++;

	return matches;
}
"""
# this function takes an array, the size of the array, and a number. The function will search the
# array for said number and return the number of occurrences. 
def searchArray(array, search_item, size) -> float:
    matches = 0
    
    for i in range(size):
        if array[i] == search_item:
            matches += 1
    
    return matches

# --- MAIN ------------------------------------------------------------------------------
# --- CHECK CLI ARGS ----------------------------
"""
    if(argc != 2) {
        cerr << "error: arguments must be: exe and filename.\n";
        exit(1);
    }
"""
# check if argv is used correctly (2 args)
if len(sys.argv) != 2:
    sys.stderr.write("error: invalid arguments (Usage: python3 dataDetails.py <file name>)")
    exit(1)

# --- INTRODUCTION ------------------------------
"""
int main(int argc, char* argv[])
{	
    cout << "This program loads an array from a file and sorts the data. It "
	     << "then outputs the size of the data, median, minimum, maximum. It "
		 << "will then prompt a to search the data for a value, and output the"
		 << " number of occurences." << endl << endl;
"""
# introduction to program
print("""This program loads an array from a file and sorts the data. It then outputs the size of
      the data, median, minimum, maximum. It will then prompt a to search the data for a value, and
      output the number of occurences.""")

# --- CONFIRMATION ------------------------------
"""
    cout << "Do you want to run this program? [Y/n]: ";
    char confirmation;
    cin >> confirmation;

    if(confirmation == 'n') {
        cout << "terminating...\n";
        exit(0);
    }
"""
confirmation = input("Do you want to run this program? [Y/n]: ")
# if declined, terminate
if confirmation == 'n':
    print("terminating...")
    exit(0)

# --- VARS --------------------------------------
"""
	double nums[MAX_SIZE];

	int size = 0;
"""
# nums declaration moved to loadArray function

size = 0
# NOTE: DATA IS TERMINATED BY -1

# --- FILE READING ------------------------------
"""
	loadArray(nums,argv[1],size);
"""
# read in values from file
nums, size = loadArray(sys.argv[1],size)

# --- PRE + POST SORT ---------------------------
"""
	cout << "PRE-SORT ARRAY (RAW)" << endl;
	printArray(nums, size);

	sortArray(nums, size);

	cout << endl << "POST-SORT ARRAY (EDITED)" << endl;
	printArray(nums, size);
"""
# pre sort
print("PRE-SORT ARRAY (RAW)")
printArray(nums,size)
# sort
sortArray(nums,size)
# post sort
print("POST-SORT ARRAY (EDITED)")
printArray(nums,size)

# --- ARRAY ACTIONS -----------------------------
"""
	cout << endl;
	cout << "Size: " << size << " values" << endl << endl;
	cout << "Median: " << median(nums,size) << endl;
	cout << "Minimum: " << minimum(nums) << endl;
	cout << "Maximum: " << maximum(nums,size) << endl;
	cout << "\nAverage: " << average(nums,size) << endl;
"""
print(f"\nSize: {size} values\n")
print("Median:",median(nums,size))
print("\nMinimmum value:",minimum(nums))
print("Maximum value:",maximum(nums,size))
print("Average value:",average(nums,size))

# --- SEARCH ------------------------------------
"""
	int numSearch;
	cout << endl << "Enter number: ";
	cin >> numSearch;

	cout << "Number of occurrences of the integer " << numSearch << ": " 
	     << searchArray(nums, numSearch, size) << endl;

	return 0;
}
"""
# variable for checking cadence of number
num_search = float(input("Enter a number: "))

# show how many occurrences 
print(f"Number of occurences of the float {num_search}: {searchArray(nums,num_search,size)}")

