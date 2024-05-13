/*
   William Wadsworth
   11.12.20
   CSC1710
   ~/csc1710/lab12
   
*/

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

void readArray(studentType array[]);

int main ()
{
   studentType students[20];
   ifstream data;
   data.open("data.txt");

   readArray(students);

   cout << "Student Name" << setw(20) << "Test Score" << setw(10) << "Grade" << endl << endl;
   for (int i = 0; i < 20; i++)
      cout << students[i].studentFName << " " << students[i].studentLName << setw(15) << students[i].testScore << setw(12) << students[i].grade << endl;





   data.close();
   return 0;
}

void readArray(studentType array[])
{
   ifstream data;

   for (int i = 0; i < 20; i++)
   {
      data >> array[i].studentFName;
      data >> array[i].studentLName;
      data >> array[i].testScore;

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








