#include <iostream>
#include <string>
using namespace std;

void strhashing()
{
   string h1 = "Educba";
   cout << "string: " << h1 << endl;
   hash<string> hashObj;
   cout << "hash value: " << hashObj(h1) << endl;
}

int main()
{
   strhashing();

   return 0;
}
