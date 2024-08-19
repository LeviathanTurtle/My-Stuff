# ARRAY MEASUREMENTS OF CENTER -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.5.2020
# Doctored: 10.25.2023
# Updated 10.25.2023: added average function
# 
# Python-ized: 3.31.2024
# Updated 8.17.2024: function decomposition and PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program loads an array from a file and sorts the data. It then outputs the size of the data,
# median, minimum, maximum. It will then prompt a to search the data for a value, and output the
# number of occurences. The end of the data in the file is noted by a value of -1.
# 
# [USAGE]:
# python3 dataDetails.py <file name>
# 
# [EXIT/TERMINATING CODES]:
# 0 - the user specified information was incorrect or the program successfully completed a full 
#     execution.
# 
# 1 - command line arguments were used incorrectly
# 
# 2 - the file was unable to be opened or created


# --- IMPORTS ---------------------------------------------------------------------------
from sys import argv, stderr, exit
from typing import List, Tuple

MAX_SIZE: int = 1000
DEBUG: bool = False

# --- FUNCTIONS -------------------------------------------------------------------------
# --- LOAD ARRAY --------------------------------
# pre-condition: filename is a valid path to a file containing float numbers, size is an integer
#                starting at 0, representing the initial size of the array
# post-condition: returns a tuple containing the list of floats and the size of the array. The file
#                 is read until either a -1 is encountered, or the maximum size (MAX_SIZE) is
#                 reached. If the file cannot be opened, the program exits with an error
def load_array(filename: str, size: int) -> Tuple[List[float], int]:
    """Load an array of floats from a file."""
    
    if DEBUG:
        print("Entering load_array...")
    
    # numbers array
    nums: List[float] = []
    size: int = 0
    
    try:
        # open user-specified file
        with open(filename,'r') as file:
            # as long as the current value is non-terminating (-1), update the size of the array
            # and get the next one
            for line in file:
                number = float(line.strip())
                if number == -1 or len(nums) == MAX_SIZE:
                    break
                
                # add the number to the list
                nums.append(number)
                # update size
                size += 1
                
    except IOError:
        stderr.write(f"error: file unable to be opened or created (provided name: {filename})")
        exit(2)

    if DEBUG:
        print("Exiting load_array.")
    return nums, size


# --- PRINT ARRAY -------------------------------
# pre-condition: numbers is a list of floats, size is the number of elements to be printed from the
#                list
# post-condition: the elements of the list up to the specified size are printed
def print_array(numbers, size):
    """Print the elements of the array."""
    
    if DEBUG:
        print("Entering print_array...")
        
    for i in range(size):
        print(f"nums[{i}] = {numbers[i]}")
    
    if DEBUG:
        print("Exiting print_array.")


# --- SORT ARRAY --------------------------------
# pre-condition: num_array is a list of floats, size is the number of elements in the list
# post-condition: the list is sorted in place in ascending order
def sort_array(num_array: List[float], size: int) -> None:
    """Sort the array in ascending order using the bubble sort algorithm."""
    
    if DEBUG:
        print("Entering sort_array...")
    
    for i in range(size-1):
        for j in range(0,size-i-1):
            # is the current value larger than the next?
            if num_array[j] > num_array[j+1]:
                # swap numbers
                num_array[j], num_array[j+1] = num_array[j+1], num_array[j]
    
    if DEBUG:
        print("Exiting sort_array.")


# --- MEDIAN ------------------------------------
# pre-condition: num_array is a list of floats, size is the number of elements in the list
# post-condition: the list is sorted in place in ascending order
def median(array: List[float], size: int) -> float:
    """Calculate the median of the sorted array."""
    
    if DEBUG:
        print("Entering median...")
        
    if size % 2 != 0:
        if DEBUG:
            print("Exiting median.")
        return array[size // 2]
    else:
        if DEBUG:
            print("Exiting median.")
        return (array[size // 2] + array[(size // 2) - 1]) / 2


# --- AVERAGE -----------------------------------
# pre-condition: array is a list of floats, size is the number of elements in the list
# post-condition: returns the average of the elements in the array
def average(array: List[float], size: int) -> float:
    """Calculate the average of the array."""
    
    if DEBUG:
        print("Entering average...")
        
    total_sum: float = sum(array[:size])
    
    if DEBUG:
        print("Exiting average.")
    return total_sum/size


# --- MINIMUM -----------------------------------
# pre-condition: array is a list of floats
# post-condition: returns the smallest value in the array
def minimum(array: List[float]) -> float:
    """Find the minimum value in the array."""
    
    return array[0]


# --- MAXIMUM -----------------------------------
# pre-condition: array is a list of floats, size is the number of elements in the list
# post-condition: returns the largest value in the array
def maximum(array: List[float], size: int) -> float:
    """Find the maximum value in the array."""
    
    return array[size-1]


# --- SEARCH ARRAY ------------------------------
# pre-condition: array is a list of floats, search_item is the float value to search for, size is
#                the number of elements in the list
# post-condition: returns the number of times search_item appears in the array
def search_array(array, search_item, size) -> float:
    """Search for occurrences of a value in the array."""
    
    return array[:size].count(search_item)


def main():
    # --- CHECK CLI ARGS ------------------------
    # check if argv is used correctly (2 args)
    if len(argv) != 2:
        stderr.write("error: invalid arguments (Usage: python3 dataDetails.py <file name>)")
        exit(1)

    # --- INTRODUCTION --------------------------
    # introduction to program
    print("""This program loads an array from a file and sorts the data. It then outputs the size
          of the data, median, minimum, maximum. It will then prompt a to search the data for a
          value, and output the number of occurences.""")

    # --- CONFIRMATION --------------------------
    confirmation = input("Do you want to run this program? [Y/n]: ")
    # if declined, terminate
    if confirmation == 'n':
        print("terminating...")
        exit(0)

    # --- VARS ----------------------------------
    size: int = 0
    # NOTE: DATA IS TERMINATED BY -1

    # --- FILE READING --------------------------
    # read in values from file
    nums, size = load_array(argv[1],size)

    # --- PRE + POST SORT -----------------------
    # pre sort
    print("PRE-SORT ARRAY (RAW)")
    print_array(nums,size)
    # sort
    sort_array(nums,size)
    # post sort
    print("POST-SORT ARRAY (EDITED)")
    print_array(nums,size)

    # --- ARRAY ACTIONS -------------------------
    print(f"""\nSize: {size} values\n\nMedian: {median(nums,size)}\n\nMinimum value: {minimum(nums)}\n
          Maximum value: {maximum(nums,size)}\nAverage value: {average(nums,size)}""")


    # --- SEARCH --------------------------------
    # variable for checking cadence of number
    num_search = float(input("Enter a number: "))

    # show how many occurrences 
    print(f"Number of occurences of the float {num_search}: {search_array(nums,num_search,size)}")


if __name__ == "__main__":
    main()