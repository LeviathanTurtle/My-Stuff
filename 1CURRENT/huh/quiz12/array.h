/*
   William Wadsworth
   CSC1710
   12.8.20
   ~/csc1710/lab11/wadsworthLab11.cpp
   make an array from a selected data file, sort, and find mean, minimum, and maximum
*/

#include <iostream>
#include <fstream>
#include <string>
using namespace std;

void loadArray(double n[], ifstream& file);
void printArray(double numbers[], int index, int size);
void sortArray(double numArray[], int index, int size);
double median(double array[], int size);
double minimum(double array[]);
double maximum(double array[], int size);
int searchArray(double array[], int searchItem, int size);
