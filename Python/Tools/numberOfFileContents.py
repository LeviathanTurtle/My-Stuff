# NUMBER OF CONTENTS IN A FILE -- V.PY
# William Wadsworth
# CSC1710
# 3.13.2024
#  
# This program reads through the contents of a file and counts how many itmes are in it.
# 
# [USAGE]:
# To run: python3 numberOfFileContents.py <filename>
# 
# [EXIT CODES]:
# 1 - incorrect CLI argument usage
# 
# 2 - input file not found 


# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <fstream>
#include <sys/time.h>
using namespace std;
"""
from time import time#, clock_gettime
import sys

# 2 args: exe file
# --- MAIN ------------------------------------------------------------------------------
# --- ARG CHECK ---------------------------------

if len(sys.argv) < 2:
    print("Usage: python3 numberOfFileContents.py filename")
    sys.exit(1)

# --- TIME BEGIN --------------------------------
"""
int main(int argc, char* argv[])
{
    struct timeval startTime, stopTime;
    double start, stop, diff;
    gettimeofday(&startTime,NULL);
"""
startTime = time()

# --- INPUT FILE + MAIN LOOP --------------------
"""
    ifstream file (argv[1]);
    
    int size=0;
    while(file.peek() != EOF) {
        if(file.peek() == '\n')
            size++;
        file.get();
    }
    file.close();
"""
inputFile = sys.argv[1]

size = 0
try:
    with open(inputFile, 'r') as file:
        for line in file:
            items = line.split()
            size += len(items)

except FileNotFoundError:
    print(f"File '{inputFile}' not found.")
    sys.exit(2)


# --- TIME END ----------------------------------
"""
    gettimeofday(&stopTime,NULL);
    start = startTime.tv_sec + (startTime.tv_usec/1000000.0);
    stop = stopTime.tv_sec + (stopTime.tv_usec/1000000.0);

    diff = stop - start;
"""
stopTime = time()
diff = stopTime - startTime

# --- END OUTPUT --------------------------------
"""
    cout << "size of file: " << size << endl;
    cout << "time elapsed: " << diff << "s" << endl;
"""
print(f"size of file: {size}")
print(f"time elapsed: {diff} s")


#    return 0;
#}
