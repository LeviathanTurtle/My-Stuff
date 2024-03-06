
#include <iostream>
#include <fstream>

using namespace std;

bool DEBUG = true;

int sum(int*, const int&);

int main(int argc, char* argv[])
{ 
    if(DEBUG)
        cout << "beginning program." << endl;

    if(DEBUG)
        cout << "opening file..." << endl;
    ifstream file (argv[1]);
    if(!file) {
        cerr << "unable to open or create file.\n";
        exit(1);
    }
    if(DEBUG)
        cout << "file opened." << endl;
    
    int K;
    file >> K;

    if(DEBUG)
        cout << "creating and loading array..." << endl;
    int* list = new int [K];
    for(int j=0; j<K; j++)
        file >> list[j];
    if(DEBUG)
        cout << "array created and loaded" << endl;

    int passes = 0;
    // ORDER:
    // 1 13
    // 18
    // 23 99

    // variable for order of desks
    int numLine = 1;
    // matrix index
    int i = 0;

    if(DEBUG)
        cout << "beginning main loop" << endl;
    while(sum(list,K) != 0) {
        // SEARCH FOR numLine IN ARRAY

        cout << list[i] << endl;
        if(list[i] == numLine) {
            cout << list[i] << " has been found,";
            list[i] = 0;
            cout << " now set to 0\n";
            i++;
            cout << "i is now " << i << endl;
        }
        else if(list[i] < numLine) {

        }
        else {
            cout << "a new pass is required\n";
            passes++;
            numLine++;
        }
    }
    if(DEBUG)
        cout << "main loop finished" << endl;


    cout << passes << endl;

    file.close();
    return 0;
}

int sum(int* list, const int& size)
{
    int sum = 0;
    
    for(int i=0; i<size; i++)
        sum += list[i];
    
    return sum;
}