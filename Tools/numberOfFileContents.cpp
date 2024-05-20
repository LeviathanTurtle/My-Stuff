/* NUMBER OF CONTENTS IN A FILE -- V.PY
 * William Wadsworth
 * CSC1710
 * 3.13.2024
 * 
 * This program reads through the contents of a file and counts how many items are in it.
*/ 

#include <iostream>
#include <fstream>
#include <sys/time.h>
using namespace std;

// 2 args: exe file
int main(int argc, char* argv[])
{
    struct timeval startTime, stopTime;
    double start, stop, diff;
    gettimeofday(&startTime,NULL);

    ifstream file (argv[1]);

    int size=0;
    while(file.peek() != EOF) {
        if(file.peek() == '\n')
            size++;
        file.get();
    }
    file.close();

    gettimeofday(&stopTime,NULL);
    start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
    stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0);

    diff = stop - start;

    cout << "size of file: " << size << endl;
    cout << "time elapsed: " << diff << "s" << endl;

    return 0;
}
