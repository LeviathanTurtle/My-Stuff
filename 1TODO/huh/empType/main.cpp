/* Author: William Wadsworth
   Date: 1.26.21
   Class: CSC-1720
   Code location: ~/csc1720/lab3/main.cpp
   
   About: 
   This program will process a file containing the information about
      the employees. MAX will indicate the maximum number that can be
      processed.  Sample data file named "employeeDB_withID":

   42 Sam 24 12345.62892
   99 Jr 43 1203.27003
   29 Mickey 52 203402
   04 Alice 43 1023.2704
   32 Mary 19 103234.35345
   84 JimBob 43 40203.378883
   54 AmySue 43 23203.3903

   The output should be formated as follows with the salary 
   rounded to two decimal places.

   ID     Name        Age       Salary
   04     Alice        43      1023.27
   29     Mickey       52    203402.00
   32     Mary         19    103234.35
   42     Sam          24     12345.63
   54     AmySue       43     23203.39
   84     JimBob       43     10203.38
   99     Jr           43      1203.27

   To compile:
      g++ -Wall main.cpp empType.cpp -o processEmployee 

   Create a data file named "employeeDB" containing the sample data 
      from above with some additions.

   NOTE: the load functions opens/closes the file employeeDB for processing.
         the dump function outputs to the screen (stdout).

   To execute:
      ./processEmployee

*/

#include<iostream>
#include<fstream>
#include<iomanip>
#include"empType.h"

using namespace std;

#define MAX 100

//prototypes
void load(empType E[], int &n);
void dump(empType E[], int n);

void exchange(empType &a, empType &b);
void shellSort(empType array[], int q);
bool shellPass(empType list[], int k, int space);

int main()
{
   empType employeeDB[MAX];
   int n;  

   // load array with employees
   load(employeeDB, n);
   // sort array
   shellSort(employeeDB, n);
   // output sorted array
   dump(employeeDB, n);

   return 0;
}

/*
   load - read in data about all employes, name, age, and salary.
   Store the data in an array to be later processed.  The data
   will be loaded into positions 0 through (num employees - 1)
   We are reading from the keyboard so I/O redirection is recommended.
   pre-condition: The empType array references an array that can
                  be loaded with the employee data.  cnt will 
                  initialy be set to 0 but will reflect the total
                  number of employees in the end.
   post-condition: The empType array will be loaded with all 
                   data found in the file or upto MAX 
   Assumption: If the employee's name can be read, we'll assume
               that their age and salary follow.

*/
void load(empType E[], int &cnt)
{
   // datafile variable
   ifstream inFile;

   // open datafile, if file not found, error
   inFile.open("employeeDB_withID");
   if(inFile.fail()) {
      cerr << "Error: Could not open the file" << endl;
      exit(1);
   }
   cnt = 0;
   string name;
   int age;
   double salary;
   int id;
   // read in first employee data, continue until EOF
   inFile >> id >> name >> age >> salary;
   while(!inFile.eof() && cnt < MAX) {
      E[cnt].setID(id);
      E[cnt].setName(name);
      E[cnt].setAge(age);
      E[cnt].setSalary(salary);
      cnt++;
      inFile >> id >> name >> age >> salary;   
   }    
   inFile.close();
}

/*
   dump - output the contents of an empType array containing n elements
          to stdout.  Consider using I/O redirection.
   pre-condition: Array E is loaded with employee data for n employees.
   post-condition: No changes will be made to the array.
*/
void dump(empType E[], int n)
{
   // heading
   cout << fixed << showpoint;
   cout << left << setw(7) << "ID";
   cout << left << setw(10) << "Name";
   cout << right << setw(5) << "Age";
   cout << right << setw(13) << "Salary" << endl;
   
   // output employee data
   for(int i=0;i<n;i++) {
      if (E[i].getID() < 10)
         cout << "0";
      cout << left << setw(7) << E[i].getID();
      cout << left << setw(10) << E[i].getName();
      cout << right << setw(5) << E[i].getAge();
      cout << right << setw(13) << setprecision(2) << E[i].getSalary();
      cout << endl;
   }
}

/*
   exchange - rotate values between two variables with a temp holder
   pre-condition: two employees from array are passed
   post-condition: two employees switch places in the array
*/
// hey hold this for me 
void exchange(empType &a, empType &b)
{
   empType c;
   c = a;
   a = b;
   b = c;
}

/*
   shellSort - while the array is not sorted, sort it via shell sort
   pre-condition: array named array is loaded with employee data with q as a 
                  space between employees in the array
   post-condition: nothing is changed in the array
*/
// 
void shellSort(empType array[], int q)
{
   int wide = q/2;
   while (wide != 0)
      if (shellPass(array, q, wide))
         wide /= 2;
}

/*
   shellPass - compare employee IDs (two at once), and call exchange method if
               they are unordered
   pre-condition: array list is loaded with employee data for k employees, and
                  space as the gap between in the array
   post-condition: if the function is sorted after calling exchange method
                   k-space number of times, return true
*/
// do the thing
bool shellPass(empType list[], int k, int space)
{
   bool alike = true;

   for (int u = 0; u < k - space; u++)
      if (list[u].getID() > list[u+space].getID())
      {
         exchange(list[u], list[u+space]);
         alike = false;
      }

   return alike;
}









/*
COMMENTS
COMMENTS
COMMENTS
COMMENTS
COMMENTS
COMMENTS
COMMENTS
COMMENTS
COMMENTS
COMMENTS
*/













