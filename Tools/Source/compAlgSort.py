# USER-SPECIFIED SORTING OF RANDOM NUMBERS
# William Wadsworth + Kolt Byers
# Created: 11.29.2021
# Doctored: 10.15.2023
# 
# Python-ized: 4.23.2024
# Updated 8.17.2024: function decomposition and PEP 8 Compliance
# 
# CSC 2342
# 
# [DESCRIPTION]:
# This program takes random values from a data file and puts them into an array. The program then
# calls user-specified sorting algorithms to sort the array from least to greatest. The program
# outputs the frequency of the swap statements and the number of elements moved.
# 
# [USAGE]:
# To run: python3 compAlgSort

# --- IMPORTS ---------------------------------------------------------------------------
from time import time
from typing import List

MAX: int = 100000
DEBUG: bool = False

# --- FUNCTIONS -------------------------------------------------------------------------
# --- BUBBLE SORT 1 -----------------------------
# pre-condition: a is a list of integers, n is the length of a, output is a list of 2 integers
# post-condition: a is sorted in ascending order, output[0] contains the frequency count of
#                 comparisons, output[1] contains the count of element movements

#   bubble1(array,        amount, results)
def bubble1(a: List[int], n: int, output: List[int]) -> None:
    """Perform bubble sort on the list `a` and update `output` with frequency and movement counts."""
   
    if DEBUG:
        print("Entering bubble1...")
      
    for _ in range(n-1):
        for j in range(n-1):
            output[0] += 1 # if frequency counter
            if a[j] > a[j+1]:
                temp = a[j]
                output[1] += 1
                a[j] = a[j+1]
                output[1] += 1
                a[j+1] = temp
                output[1] += 1 # elements moved  counter
   
    if DEBUG:
        print("Exiting bubble1.")

# --- BUBBLE SORT 2 -----------------------------
# pre-condition: a is a list of integers, n is the length of a, output is a list of 4 integers
# post-condition: a is sorted in ascending order, output[2] contains the frequency count of
#                 comparisons, output[3] contains the count of element movements

#   bubble2(array,        amount, results)
def bubble2(a: List[int], n: int, output: List[int]) -> None:
    """Perform bubble sort (variant) on the list `a` and update `output` with frequency and movement counts."""
   
    if DEBUG:
        print("Entering bubble2...")
      
    for _ in range(n-1):
        for j in range(n-1):
            output[2] += 1
            if a[j] > a[j+1]:
                temp = a[j]
                output[3] += 1
                a[j] = a[j+1]
                output[3] += 1
                a[j+1] = temp
                output[3] += 1
   
    if DEBUG:
        print("Exiting bubble2.")

# --- MERGE SORT --------------------------------
# pre-condition: a is a list of integers, n is the length of a, output is a list of 4 integers
# post-condition: a is sorted in ascending order, output[4] contains the frequency recursive calls,
#                 output[5] contains the count of element movements
def merge_sort(a: List[int], output: List[int], start: int, end: int) -> None:
    """Perform merge sort on the list `a` and update `output` with frequency and movement counts."""
   
    if DEBUG:
        print("Entering merge_sort...")
      
    output[4] += 1
    if start < end:
        middle = start + (end-start) // 2
        merge_sort(a, output, start, middle)
        merge_sort(a, output, middle+1, end)
        merge(a, output, start, middle, end)
   
    if DEBUG:
        print("Exiting merge_sort.")

# --- MERGE -------------------------------------
# pre-condition: m is a list of integers, output is a list of 6 integers, front, midL, and back are
#                indices defining the segments to merge
# post-condition: the segment of m from front to back is sorted, output[5] contains the count of
#                 element movements
def merge(m: List[int], output: List[int], front: int, midL: int, back: int) -> None:
    """Merge two sorted segments of list `m` into a single sorted segment."""
   
    if DEBUG:
        print("Entering merge...")
      
    t = [MAX] * (back - front + 1)
    l1, r1 = front, midL
    l2, r2 = midL + 1, back
    n = 0
   
    while l1 <= r1 and l2 <= r2:
        if m[l1] < m[l2]:
            t[n] = m[l1]
            l1 += 1
        else:
            t[n] = m[l2]
            l2 += 1
        n += 1
        output[5] += 1
    
    while l1 <= r1:
        t[n] = m[l1]
        n += 1
        l1 += 1
        output[5] += 1
    
    while l2 <= r2:
        t[n] = m[l2]
        n += 1
        l2 += 1
        output[5] += 1
    
    for i in range(front, back + 1):
        m[i] = t[i - front]
        output[5] += 1
   
    if DEBUG:
        print("Exiting merge.")

# --- LOAD SORT ---------------------------------
# asks user for sorting algorithm, calls designated function, outputs everything except time
# pre-condition: data is a list of integers, results is a list of 6 integers, sortAmt is the number
#                of elements in data, sortNum is an integer specifying the sorting algorithm (1, 2,
#                or 3).
# post-condition: results is updated with counts based on the sort performed
def load_sort(data: List[int], results: List[int], sortAmt: int, sortNum: int) -> None:
    """Load data and perform the specified sort, updating results with counts."""
    
    if DEBUG:
        print("Entering load_sort...")
      
    # input validation
    while sortNum < 1 and sortNum > 3:
        sortNum = int(input(f"error: invalid sort method number ({sortNum}): "))
      
    # run sort, store selected store frequency in variable
    methFreq: int = 0
    # same thing with elements moved
    elmtMov: int = 0
   
    # call bubble sort 1 if specified
    if sortNum == 1:
        bubble1(data,sortAmt,results)
        methFreq = results[0]
        elmtMov = results[1]
    # call bubble sort 2 if specified
    elif sortNum == 2:
        bubble2(data,sortAmt,results)
        methFreq = results[2]
        elmtMov = results[3]
    # call merge sort if specified
    elif sortNum == 3:
        start: int = 0
        end: int = sortAmt-1
      
        merge_sort(data,results,start,end)
        methFreq = results[4]-1
        elmtMov = results[5]
      
    # output results
    if sortNum == 3:
        # results output for merge
        print(f"Results:\n\n  n: {sortAmt}\n  sort: {sortNum}\n  recursive calls: {methFreq}\n  elements moved: {elmtMov}\n")
    else:
        # results output for bubble
        print(f"Results:\n\n  n: {sortAmt}\n  sort: {sortNum}\n  if frequency: {methFreq}\n  elements moved: {elmtMov}\n")

    if DEBUG:
       print("Exiting load_sort.")

# --- LOAD ARRAY --------------------------------
# load array definition: loads values from data file
# pre-condition: dataFile is the path to the file to write, array is a list of integers, amount is
#                the number of elements in array 
# post-condition: the file dataFile contains the integers from array
def load_array(dataFile: str, array: List[int], amount: int) -> None:
    """Load an array of integers into a file."""
    
    if DEBUG:
        print("Entering load_array...")
      
    with open(dataFile, 'w') as file:
        for i in range(amount):
            file.write(f"{array[i]}\n")
   
    print("Data file successfully loaded")
   
    if DEBUG:
        print("Exiting load_array.")


def main():
    # query dataset, open file
    dataResp = input("Enter the dataset: ")
    print("\n")
      
    # --- CREATE DATASET ------------------------
    # query number of processed items
    sortAmt = int(input("Enter the number of random numbers to process [0, 100,000]: "))
    sortAmt = min(max(sortAmt, 0), MAX)

    data = [0] * sortAmt

    # load array with data
    load_array(dataResp,data,sortAmt)

    # --- SORT, RESULTS SETUP -------------------
    # query sort method number
    sortNum = int(input("Enter the type of sort (1=bubble1, 2=bubble2, 3=merge): "))

    # results, put in array for ease of use
    results = [0] * 6
    # results[0] = bubble1 if frequency
    #        [1] = bubble1 elements moved
    #        [2] = bubble2 if frequency
    #        [3] = bubble2 elements moved
    #        [4] = mergeSort recursive calls
    #        [5] = mergeSort elements moved

    # --- TIMEVAL -------------------------------
    startTime = time()
    load_sort(data,results,sortAmt,sortNum)
    stopTime = time()

    diff = stopTime - startTime

    # --- OUTPUT --------------------------------
    print(f"  time: {diff:.5f} seconds")

    # choice to output sorted array
    yn = input("Output the sorted list? [y/n]: ").strip().lower()
    if yn == 'y':
        for i in range(sortAmt):
            print(f"{data[i]} ")
    print("\n")


if __name__ == "__main__":
    main()