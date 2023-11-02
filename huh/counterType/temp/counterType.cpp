
#include "counterType.h"
#include <iostream>
using namespace std;

/*
   initializeCounter - initialize variable counter
   pre-condition: variable counter is defined
   post-condition: counter will be set to 0
*/
void counterType::initializeCounter()
{
   counter = 0;
}

/*
   setCounter - initialize counter, ensure counter is not initialized below 0
   pre-condition: integer c must be declared and initialized; counter is 
                  delcared
   post-condition: counter will be set to 0 or c
*/
void counterType::setCounter(int c)
{
   if (c < 0)
      counter = 0;
   else
      counter = c;
}

/*
   getCounter - return private variable counter
   pre-condition: counter variable is declared and initialized
   post-condition: variable is returned
*/
int counterType::getCounter() const
{
   return counter;
}

/*
   incrementCounter - increase counter by 1
   pre-condition: counter variable is declared and initialized
   post-condition: none
*/
void counterType::incrementCounter()
{
   counter++;
}

/*
   decrementCounter - decrease counter by 1, reset to 0 if difference is < 0
   pre-condition: counter variable is declared and initialized
   post-condition: none
*/
void counterType::decrementCounter()
{
   if (counter == 0)
      counter = 0;
   else
      counter--;
}

/*
   displayCounter - output counter value
   pre-condition: counter variable is declared and initialized
   post-condition: none
*/
void counterType::displayCounter() const
{
   cout << counter;
}

/*
   counterType - initialize counter, ensure counter is not initialized below 0
   pre-condition: integet t is declared and initialized; counter is declared
   post-condition: counter will be set to 0 or t
*/
counterType::counterType(int t)
{
   if (t < 0)
      counter = 0;
   else
      counter = t;
}

/*
   counterType - default constructor, instantiate variable counter
   pre-condition: counter is declared
   post-condition: counter will be set to 0
*/
counterType::counterType()
{
   counter = 0;
}

