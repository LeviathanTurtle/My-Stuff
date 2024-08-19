# NUMBER OF CONTENTS IN A FILE -- V.PY
# William Wadsworth
# CSC1710
# 3.13.2024
# Updated 8.17.2024: function decomposition and PEP 8 Compliance
#  
# This program reads through the contents of a file and counts how many items are in it.
# 
# [USAGE]:
# To run: python3 numberOfFileContents.py <filename>
# 
# [EXIT CODES]:
# 1 - incorrect CLI argument usage
# 
# 2 - input file not found 


# --- IMPORTS ---------------------------------------------------------------------------
from time import time
from sys import argv, stderr, exit

DEBUG: bool = False


# pre-condition: filename is the name of the file to be read
# post-condition: returns the total number of words in the file
def count_words_in_file(filename: str) -> int:
    """Counts the number of words in the specified file."""
    
    if DEBUG:
        print("Entering count_words_in_file...")
    
    word_count: int = 0
    
    try:
        with open(filename, 'r') as file:
            for line in file:
                word_count += len(line.split())
    except FileNotFoundError:
        stderr.write(f"Error: File '{filename}' not found.\n")
        exit(2)
    
    if DEBUG:
        print("Exiting count_words_in_file.")
    return word_count


def main():
    # --- ARG CHECK -----------------------------
    # 2 args: exe file
    if len(argv) != 2:
        stderr.write("Usage: python3 numberOfFileContents.py <filename>")
        exit(1)

    # --- INPUT FILE ----------------------------
    filename = argv[1]
    
    # --- TIME BEGIN ----------------------------
    startTime = time()

    word_count = count_words_in_file(filename)

    # --- TIME END ------------------------------
    stopTime = time()

    # --- END OUTPUT ----------------------------
    print(f"size of file: {word_count}")
    print(f"time elapsed: {stopTime-startTime}s")


if __name__ == "__main__":
    main()