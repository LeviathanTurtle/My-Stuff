/*
   William Wadsworth
   Professor Shore
   10.24.21
   CSC
   ~/discrete/prog1
   This program uses I/O redirection to read in strings into a universal set, then create subsets
 */

#include <iostream>
#include <string>
using namespace std;

struct element
{
   string name;
   int bitNum = 0;
};

void print(element[], int);
//void deallocate(element, int);
//void load(element[], element[], int);

int main()
{
   //===============================================================
   // VARIABLES

   element uSet[32];
   element aSet[32];
   element bSet[32];

   // can I not do int uSetSize = aSetSize = bSetSize = 0?
   int uSetSize = 0, aSetSize = 0, bSetSize = 0;

   int p = 1;

   //===============================================================
   // READING UNIVERSAL
    
   // read in data from I/O, throw into uSet, count how many are in A and B
   int i = 0;
   while (cin.peek() != '\n')
   {
      cin >> uSet[i].name;
      uSet[i].bitNum = p;
      uSetSize++;

      if (uSet[i].name == "apple" || uSet[i].name == "pear" || uSet[i].name == "grape" || uSet[i].name == "cherry")
      {
         //aSet[i].name = uSet[i].name;
         //aSet[i].bit = uSet[i].bit;
         aSetSize++;
      }
      if (uSet[i].name == "pear" || uSet[i].name == "orange" || uSet[i].name == "cherry")
      {
         //bSet[i].name = uSet[i].name;
         //bSet[i].bit = uSet[i].bit;
         bSetSize++;
      }

      p *= 2;
      i++;
   }

   //===============================================================
   // JUNKYARD - I HATE POINTERS they can burn forever
   /*
   const int qwerty = 5;
   element oweuh[qwerty];
   element aSet[aSetSize];
   element bSet[bSetSize];

   // I tried dynamic allocation. Didn't work
   element* foo = new element[aSetSize];
   element** aSet = new element*[aSetSize];
   element* faa = new element[bSetSize];
   element** bSet = new element*[bSetSize];
   */
   //===============================================================
   // LOADING A AND B
   const string in = "1";
   const string out = "0";

   // load A
    
   string aBit;
   int l = 0;
   for (int i = 0; i < uSetSize; i++)
   {  
      if (uSet[i].name == "apple" || uSet[i].name == "pear" || uSet[i].name == "grape" || uSet[i].name == "cherry")
      {
         aSet[l].name = uSet[i].name;
         aSet[l].bitNum = uSet[i].bitNum;
         aBit.append(in);
         l++;
      }
      else
         aBit.append(out);
   }
   //load(aSet, uSet, uSetSize);

   // load B
    
   string bBit;
   int t = 0;
   for (int i = 0; i < uSetSize; i++)
   {
      if (uSet[i].name == "orange" || uSet[i].name == "pear" || uSet[i].name == "cherry")
      {
         bSet[t].name = uSet[i].name;
         bSet[t].bitNum = uSet[i].bitNum; // I'm dumb - I used uSet[t]
         bBit.append(in);
         t++;
      }
      else
         bBit.append(out);
   }
   //load(bSet, uSet, uSetSize);

   //===============================================================
   // OUTPUT AND TESTING

   // base output
   cout << "UNIVERSAL SET" << endl << endl;
   print(uSet, uSetSize);
   // please work
   cout << "A SET" << endl << endl;
   print(aSet, aSetSize);

   cout << "B SET" << endl << endl;
   print(bSet, bSetSize);

   // operations + output of outcome
   // A U B
   cout << "A union B: " << aBit & bBit << endl;

   // A compliment
   cout << "A compliment: " << ~aBit << endl;

   // A intersect B
   cout << "A intersect B: " << aBit | bBit << endl;

   // A - B
   

   // A xor B
   cout << "A xor B: " << aBit ^ bBit << endl;

   // I give up

   //===============================================================
   // MORE JUNK/END PROGRAM

   //deallocate(aSet[], aSetSize);
   //delete[] foo;
   //deallocate(bSet, bSetSize);
   //delete[] faa;
    
   return 0;
}

// CAN YOU PRINT THE RIGHT OUTPUT PLEASE?!?
void print(element s[], int size)
{
   cout << "Set size: " << size << endl << "{";
   // output names
   for (int j = 0; j < size; j++)
   {
      cout << s[j].name;
      if (j + 1 < size)
         cout << ", ";
   }
   cout << "}" << endl;

   //cout << endl;

   // output bit numbers
   /*
   for (int m = 0; m < size; m++)
      cout << s[m].bit << " ";
   cout << endl << endl << endl;*/

   // I wish I could remember how to align these neatly ... 
   // oh well this is here if you want it
}

/*
void deallocate(element set, int length)
{
   for (int i = 0; i < length; i++)
   {
      delete[] set[i].name;
      delete[] set[i].bit;
   }
}*/

// doesn't work, A and B are different :/
/*
void load(element array[], element univ[], int size)
{
   int g = 0;
   for (int i = 0; i < size; i++)
   {
      if (univ[i].name == "apple" || univ[i].name == "pear" || univ[i].name == "grape" || univ[i].name == "cherry")
      {
         array[g].name = univ[g].name;
         array[g].bit = univ[g].bit;
         g++;
      }
   }
}*/

/*
   after multiple redos, headbangings and headaches I give up
 */
