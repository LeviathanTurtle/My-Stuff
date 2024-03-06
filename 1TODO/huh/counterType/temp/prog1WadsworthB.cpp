/* 
   Author: William Wadsworth
   Date: 1.28.21
   Class: CSC-1720
   Code location: ~/csc1720/prog1/prog1WadsworthB.cpp

   About:
   This (B-level) program will open a data file, and count the number of characters, 
      including spaces, bangs, question marks, and periods. It will also keep 
      track of the number of vowels and sentences.

   The output should be formated as follows:

   Passage from [datafile]:
       Number of characters: XXX
       Number of vowels: XXX
       Number of sentences: XXX

   To compile:
      g++ -Wall prog1WadsworthB.cpp -o textCounter 

   To execute:
      ./textCounter
*/

#include "counterType.h"
#include "counterType.cpp"
#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

int main()
{
   // define datafile
   ifstream dFile;

   // open user-specified datafile, includes input validation
   string resp;
   cout << "What file would you like to open: ";
   cin >> resp;
    
   dFile.open(resp);
   if (dFile.fail())
   {
      cerr << "Error: could not open file" << endl;
      return 0;
   }

   // constructor for characters, initialize to 0 with default constructor
   counterType chars;
   // constructor for vowels, initialize to 0 with default constructor
   counterType vows;
   // constructor for sentences, initialize to 0 with default constructor
   counterType sent;

   // letter variable for reading in characters
   char letter;
   // read in first character, including whitespaces
   dFile >> noskipws >> letter;
   while (!dFile.eof())
   {
      // is the character a vowel (upper or lower case)
      if (letter == 'a' || letter == 'A' || letter == 'e' || letter == 'E' || letter == 'i' || letter == 'I' || letter == 'o' || letter == 'O' || letter == 'u' || letter == 'U')
      {
         // increment counter for vowels and total characters
         vows.incrementCounter();
         chars.incrementCounter();
      }
      else if (letter == '.' || letter == '!' || letter == '?')
      {
         // increment counter for sentences and total characters
         sent.incrementCounter();
         chars.incrementCounter();
      }
      else
      {
         // increment counter for total characters
         chars.incrementCounter();
      }

      // read in next character, including whitespaces
      dFile >> noskipws >> letter;
   }

   // heading, display datafile
   cout << "Passage from " << resp << ":" << endl;
   // display total number of characters
   cout << "   " << "Number of characters: ";
   chars.displayCounter();
   cout << endl;

   // display total number of vowels
   cout << "   " << "Number of vowels: ";
   vows.displayCounter();
   cout << endl;

   // display total number of sentences
   cout << "   " << "Number of sentences: ";
   sent.displayCounter();
   cout << endl;

   // close datafile, end program
   dFile.close();
   return 0;
}


