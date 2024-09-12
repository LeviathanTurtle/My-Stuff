#!/bin/bash

# build C++ projects
mkdir -p build
cd build
cmake ..
make

# go back to the root directory
cd ..

# create standalone executables for Python scripts in each directory
pyinstaller --onefile Calculators/Source/calculate.py --distpath Calculators/Dist
pyinstaller --onefile Games/Source/play.py --distpath Games/Dist
pyinstaller --onefile Tools/Source/utility.py --distpath Tools/Dist

echo "Build process completed."
