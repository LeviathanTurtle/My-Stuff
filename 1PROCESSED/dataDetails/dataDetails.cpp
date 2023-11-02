/*
 * William Wadsworth
 * CSC1710
 * Created: 11.5.2020
 * Doctored: 10.25.2023
 * Updated: 10.25.2023 -- added average function
 * ~/csc1710/lab11/wadsworthLab11.cpp
 * 
 * 
 * [DESCRIPTION]:
 * This program loads an array from a file and sorts the data. It then outputs
 * the size of the data, median, minimum, maximum. It will then prompt a to
 * search the data for a value, and output the number of occurences. The end of
 * the data in the file is noted by a value of -1.
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ dataDetails.cpp -Wall -o dataDetails
 *
 * To run (2 args):
 *     ./dataDetails <file name>
 * 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - the user specified information was incorrect or the program successfully
 *     completed a full execution.
 * 
 * 1 - command line arguments were used incorrectly
 * 
 * 2 - the file was unable to be opened or created
*/

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

#define MAX_SIZE 1000

// function prototypes
void loadArray(double[], const char*, const int&);
void printArray(double[], const int&);
void sortArray(double[], const int&);
double median(double[], const int&);
double average(double[], const int&);
double minimum(double[]);
double maximum(double[], const int&);
int searchArray(double[], const int&, const int&);

int main(int argc, char* argv[])
{	
	// introduction to program
    cout << "This program loads an array from a file and sorts the data. It "
	     << "then outputs the size of the data, median, minimum, maximum. It "
		 << "will then prompt a to search the data for a value, and output the"
		 << " number of occurences." << endl << endl;

	// confirm
    cout << "Do you want to run this program? [Y/n]: ";
    char confirmation;
    cin >> confirmation;
    // if declined, terminate
    if(confirmation == 'n') {
        cout << "terminating...\n";
        exit(0);
    }

	// check if argv is used correctly (2 args)
    if(argc != 2) {
        cerr << "error: arguments must be: exe and filename.\n";
        exit(1);
    }

	// numbers array
	double nums[MAX_SIZE];
	// define size, initialize to 0
	// DATA IS TERMINATED BY -1
	int size = 0;

	// read in values from file
	loadArray(nums,argv[1],size);

	// PRE SORT
	cout << "PRE-SORT ARRAY (RAW)" << endl;
	printArray(nums, size);

	// SORT
	sortArray(nums, size);

	// POST SORT
	cout << endl << "POST-SORT ARRAY (EDITED)" << endl;
	printArray(nums, size);

	// SIZE, MEDIAN, MINIMUM, MAXIMUM
	cout << endl;
	cout << "Size: " << size << " values" << endl << endl;
	cout << "Median: " << median(nums,size) << endl;
	cout << "Minimum: " << minimum(nums) << endl;
	cout << "Maximum: " << maximum(nums,size) << endl;
	cout << "\nAverage: " << average(nums,size) << endl;

	// variable for checking cadence of number
	int numSearch;
	cout << endl << "Enter number: ";
	cin >> numSearch;

	// show how many occurrences 
	cout << "Number of occurrences of the integer " << numSearch << ": " 
	     << searchArray(nums, numSearch, size) << endl;

	// end program
	return 0;
}

// this function creates an input file of a name that is passed as a parameter
// from main. If the file was opened successfully, it reads in the data while
// keeping note of the size of the array. The file is closed when done.
void loadArray(double nums[], const char* filename, int& size)
{
    // data variable, open user-specified file
	ifstream dataFile (filename);
	// check file
	if(!dataFile) {
		cerr << "error: file unable to be opened or created (provided name: "
		     << filename << ").\n";
		exit(2);
	}
	
	// array index
	int index = 0;

	// read in first value
    dataFile >> nums[index];
	// as long as the current value is non-terminating (-1), update the size 
	// of the array and get the next one
    while (nums[index] != -1) {
        size++;
		index++;
        dataFile >> nums[index];
    }

	// close file
	dataFile.close();
}

// all functions use const int& for all parameters that are not the array to
// ensure they are not accidentally edited and there are no extra copies in
// memory.

// this function takes an array and its size as parameters, and outputs each
// value in said array along with its location.
void printArray(double numbers[], const int& size)
{
	for(int i=0; i<size; i++)
		cout << "nums[" << i << "] = " << numbers[i] << endl;
}

// this function takes an array and its size as parameters and then sorts the 
// array.
void sortArray(double numArray[], const int& size)
{
	// temp variable for swap
	int hold;

	for (int pass = 1; pass < size; pass++)
		for (int i = 0; i < size - pass; i++)
			// is the current value larger than the next?
			if (numArray[i] > numArray[i+1]) {
				// swap numbers
				hold = numArray[i];
				numArray[i] = numArray[i+1];
				numArray[i+1] = hold;
			}
}

// this function takes an array and its size as parameters. The function will 
// then calculate and return the median value along with its location in the 
// array. The function assumes the array is already sorted.
double median(double array[], const int& size)
{
	double med;
	
	// if the size is odd, pick the middle value
	if (size % 2 != 0) {
		med = array[size/2];
		cout << "nums[" << size/2 << "] = ";
	}
	// if the size is even, average the two middle values
	else {
		cout << "(nums[" << size/2 << "] + nums[" << (size/2)-1 << "]) / 2 = ";
		med = (array[size/2] + array[(size/2)-1]) / 2;
	}

	return med;
}

// this function takes an array and its size as parameters. The function will
// then calculate and return the average of the values in the array.
double average(double array[], const int& size)
{
	double sum = 0;

	for(int i=0; i<size; i++)
		sum += array[i];

	return sum/size;
}

// this function takes an array and its size as parameters. The function will 
// then return the lowest value and its location in the array. The function
// assumes the array is already sorted.
double minimum(double array[])
{
	cout << "nums[0] = ";

	return array[0];
}

// this function takes an array and its size as parameters. The function will 
// then return the largest value and its location in the array. The function
// assumes the array is already sorted.
double maximum(double array[], const int& size)
{
	//double max;
	cout << "nums[" << size-1 << "] = ";

	return array[size-1];
}

// this function takes an array, the size of the array, and a number. The 
// function will search the array for said number and return the number of 
// occurrences. 
int searchArray(double array[], const int& searchItem, const int& size)
{
	int matches = 0;

	for(int i=0; i<size; i++)
		if (array[i] == searchItem)
			matches++;

	return matches;
}
