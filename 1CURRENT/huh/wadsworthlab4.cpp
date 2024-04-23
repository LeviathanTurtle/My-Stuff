/*
   William Wadsworth
   10 September 2020
   CSC1710-02
   ~/csc1710/lab4/lab4.cpp

   Limited calculator with 5 given variables
*/

#include <iostream>
#include <iomanip>
using namespace std;
int main()
{
   //declare integers and doubles
   int p=17, q=4, r=3;
   double j=3.00, k=5.0;

   //include showpoint for outputs
   cout << showpoint;

   //calculate, display outputs
   cout << "p-q*r = " << p-q*r << endl;
   cout << "p/r = " << p/r << endl;
   cout << "p%r = " << p%r << endl;
   cout << "p/q/r = " << p/q/r << endl;
   cout << "q+r%p = " << q+r%p << endl;
   cout << "q*j/p = " << q*j/p << endl;
   cout << "p/q/j = " << p/q/j << endl;
   cout << "p/j/q = " << p/j/q << endl;
   cout << "k/=r/q = " << (k/=r)/q << endl;
   cout << "j/-q = " << k/-q << endl;
   cout << ".5*p*r = " << .5*p*r << endl;
}
