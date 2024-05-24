/* NUMBER OF CONTENTS IN A FILE
 * William Wadsworth
 * CSC1710
 * 3.13.2024
 * 
 * [DESCRIPTION]:
 * This program reads through the contents of a file and counts how many items are in it. The
 * binary was last compiled on 5.24.2024.
 * 
 * [USAGE]:
 * To compile: g++ numberOfFileContents.cpp -Wall -o <exe name>
 * To run:     ./<exe name> <filename>
*/ 

#include <iostream>
#include <fstream>
#include <sys/time.h>
using namespace std;

// 2 args: exe file
int main(int argc, char* argv[])
{
    // time vars
    struct timeval startTime, stopTime;
    double start, stop, diff;
    // start time
    gettimeofday(&startTime,NULL);

    // create file
    ifstream file (argv[1]);

    // var to keep track of the size
    int size=0;

    // while there is input
    while(file.peek() != EOF) {
        // as long as the next item is not a newline, get it
        if(file.peek() == '\n')
            size++;
        file.get();
    }
    file.close();

    // stop time
    gettimeofday(&stopTime,NULL);
    // calculate time diff
    start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
    stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0);

    diff = stop - start;

    cout << "size of file: " << size << endl;
    cout << "time elapsed: " << diff << "s" << endl;

    return 0;
}
