/* 
   Author:
   Date:
   Class: CSC-1720
   Code location:
   
   About: 
*/

#include"empType.h"

void empType::setName(string iname)
{
   name = iname;
}

void empType::setAge(int iage)
{
   age = iage;
}


void empType::setSalary(double isalary)
{
   salary = isalary;
}

void empType::setID(int iid)
{
   id = iid;
}

string empType::getName(void) const
{
   return name;
}


int empType::getAge(void) const
{
   return age;
}


double empType::getSalary(void) const
{
   return salary;
}

int empType::getID(void) const
{
   return id;
}

empType::empType(string name, int age, int salary, int id)
{
   if (name == " ")
      cerr << "Error: no name" << endl;
   else if (age <= 18)
      cerr << "An employee is too young" << endl;
   else if (salary < 0)
      salary = 0;
   else if (id <= 0)
      cerr << "An employee has an invalid ID" << endl;
   else 
      cerr << "Error" << endl;
}

empType::empType()
{
   name = " ";
   age = 0;
   salary = 0;
   id = 0;
}
