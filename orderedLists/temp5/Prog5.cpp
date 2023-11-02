/* William Wadsworth
 * CSC1720
 * main program for Program 5
 */

#include <iostream>
#include <fstream>
#include "linkedlist.h"
#include "orderedLinkedList.h"
#include "stuType.h"
using namespace std;

int main()
{
	ifstream data1, data2;
	//orderedLinkedList<stuType<Type>> 
	stuType<Type> list1, list2;

	// open datafile
	cout << "opening first file..." << endl;
	data1.open("list1.txt");
	cout << "file opened." << endl;
	cout << "reading in files..." << endl;
	// load list with data from file
	list1.load(data1);
	if (list1.length() > 0)
		cout << "list loaded." << endl;
	data1.close();

	// repeat with second file and second list
	cout << "opening second file..." << endl;
	data2.open("list2.txt");
	cout << "file opened." << endl;
	cout << "reading in files..." << endl;
	list2.load(data2);
	if (list2.length() > 0)
		cout << "list loaded." << endl;
	data2.close();

	// merge lists, print combined list
	list1.merge(list2);
	list1.print();

	// find average gpa
	cout << "Average GPA: " << list1.avgGPA() << endl;

	return 0;
}

