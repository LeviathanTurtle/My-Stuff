#!/bin/bash

# loop through each .py file in the current directory
for file in Source/*.py; do
    # get filename without extension
    filename=$(basename "$file")
    program="${filename%.*}"

    # compile
    pyinstaller -F "$file"

    # move binary to current dir
    mv "dist/$program" "./$program-py.exe"
done

# remove all extra folders, spec file
rm -rf build dist *.spec