#!/bin/bash

# loop through each .cpp file in the current directory
for file in Source/*.cpp; do
    # get filename without extension
    filename=$(basename "$file")
    program="${filename%.*}"

    # compile
    g++ "$file" -Wall -o "./$program-cpp.exe"
done
