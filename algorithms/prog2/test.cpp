#include <iostream>
using namespace std;

int main() {

   int a = 0;

   cout << "a = " << a << endl;

   for(int i = 0; i < 10; i++, a++)
      cout << "test" << endl;

   cout << "a = " << a << endl;

   return 0;
}
