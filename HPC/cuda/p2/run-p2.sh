#!/bin/bash 
# 
# The mult_v1 program takes the name provided and adds -device-tileWidth-width
# to form the name of the answer file. 
#
# Example: If the program is named mult_v1 an executed with mult_v1 32 
# input-A-1024 input-B-1024 ans, it uses tile width=32, matrix width=1024. The
# output file name argument of “ans” the mult_v1 creates an output file named
# ans-32-1024. 
# 
# Also note that mult_v1 can output std_out the times which are captured in 
# avgTime and echo'ed to the display.

# shore script

: '
if [ "$#" -eq 1 ] 
then 
    device=$1 
    run="1" 
    for width in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
    do 
        for tile_width in 4 8 16 32 
        do 
            avgTime=`mult_v1 $tile_width input-A-$width input-B-$width ans` 
            numCells=`expr $width \* $width` 
            echo $run $width $tile_width $numCells $avgTime 
        done 
        run=`expr $run + 1` 
    done 
else 
    echo "Usage: cudaRun <device number>" 
fi
'

# my script

# check if CL args = 1
if [ "$#" -eq 1 ] 
then
    device=$1
    run="1"

    # repeat for every matrix size
    for width in 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384
    do
        # repeat for every tile width
        for tile_width in 4 8 16 32
        do
            # output current execution specification
            printf "run: $run, size: $width tile: $tile_width :\n"
            # main/cuda/kernel times stored here
            avgTime=`./p2 $tile_width TestFiles/input-$width TestFiles/input-$width Output/ans-$tile_width-$width`
            printf "avg time: $avgTime"
            printf "\n\n"
        done
        run=`expr $run + 1`
    done
else
    echo "Usage: run-p1.sh <device number>" 
fi

