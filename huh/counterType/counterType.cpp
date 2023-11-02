/*

*/

#include "counterType.h"
#include <iostream>
using namespace std;

void counterType::initializeCounter()
{
   counter = 0;
}

void counterType::setCounter(int c)
{
   if (c < 0)
      counter = 0;
   else
      counter = c;
}

int counterType::getCounter() const
{
   return counter;
}

void counterType::incrementCounter()
{
   counter++;
}

void counterType::decrementCounter()
{
   if (counter == 0)
      counter = 0;
   else
      counter--;
}

void counterType::displayCounter() const
{
   cout << counter;
}

counterType::counterType(int t)
{
   if (t < 0)
      counter = 0;
   else
      counter = t;
}

counterType::counterType()
{
   counter = 0;
}
