#!/bin/bash

# This script assumes max-master.cpp is compiled as "max-master-cpp" and is in the same directory

for i in {1..20..1}
do
    ./max-master-cpp 1000 10
done
