# GOLF -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.18.2020
# Doctored: 11.2.2023
# 
# Python-ized: 3.30.2024
# Updated 8.17.2024: function decomposition and PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program analyzes a file consisting of golf data (file structure below). It finds and outputs
# the player with the lowest number of strokes for the game. The number of players and holes can be
# different than the number of players and holes in the input file, that can be adjusted at runtime.
#
# [USAGE]:
# python3 golfAnalysis.py <number of players> <number of holes> <input file>
# 
# [DATA FILE STRUCTURE]:
# <pars for hole 1> <pars for hole 2> ... <pars for hole n>
# <player name 1>
# <player name 2>
# ...
# <player name m>
# <P1 strokes for hole1> <P2 strokes for hole1> ... <Pm strokes for hole1>
# <P1 strokes for hole2> <P2 strokes for hole2> ... <Pm strokes for hole2>
# ...
# <P1 strokes for holen> <P2 strokes for holen> ... <Pm strokes for holen>
# 
# where:
#   n = number of holes 
#   m = number of players 
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed full execution
# 
# 1 - CLI args used incorrectly
# 
# 2 - file unable to be opened or created

# --- IMPORTS ---------------------------------------------------------------------------
from sys import argv, stderr, exit
from typing import List

DEBUG: bool = False

# --- FUNCTIONS -------------------------------------------------------------------------
# --- MAKE HOLES --------------------------------
# pre-condition: array is a list of integers, size is the number of holes
# post-condition: the array is filled with integers from 1 to size
def make_holes(array: List[int], size: int) -> None:
    """Populate an array with hole numbers starting from 1."""
    
    if DEBUG:
        print("Entering make_holes...")
        
    for i in range(size):
        array[i] = i+1
    
    if DEBUG:
        print("Exiting make_holes.")

# --- PRINT LINE --------------------------------
# pre-condition: player_num is the index of the player, num_holes is the number of holes in the
#                game, players is a list of player names, data is a 2D list where each sublist
#                contains the strokes for each hole
# post-condition: outputs the player's name followed by their strokes per hole
def print_line(player_num: int, num_holes: int, players: List[str], data: List[List[int]]) -> None:
    """Print the strokes per hole for a given player."""
    
    if DEBUG:
        print("Entering print_line...")
        
    print(f"{players[player_num]:<9}", end="")
    
    for i in range(num_holes):
        print(f"{data[player_num][i]:<3}", end="")
    print()
    
    if DEBUG:
        print("Exiting print_line.")

# --- CALCULATE SUM -----------------------------
# pre-condition: player_num is the index of the player, hole_count is the number of holes in the
#                game, data is a 2D list where each sublist contains the strokes for each hole
# post-condition: returns the total strokes for the given player
def calculate_sum(player_num: int, hole_count: int, data: List[List[int]]) -> int:
    """Calculate the total strokes for a player."""
    
    if DEBUG:
        print("Entering calculate_sum...")
        
    # calculate number of strokes in the game
    total_strokes: int = sum(data[player_num][:hole_count])
    
    if DEBUG:
        print("Exiting calculate_sum.")
    return total_strokes


def main():
    # --- CHECK CLI ARGS ------------------------
    # check CLI args
    if len(argv) != 4:
        stderr.write("Usage: python3 golfAnalysis.py <number of players> <number of holes> <input file>")
        exit(1)

    # in order of file
    player_count = int(argv[1])
    hole_count = int(argv[2])

    # --- DATA ARRAYS -------------------------------
    # array for pars (of holes)
    pars: List[int] = [0] * hole_count

    # array for player names
    names: List[str] = [""] * player_count

    # array for player strokes per hole
    # x -> players
    # y -> holes
    results: List[List[int]] = [[0 for _ in range(hole_count)] for _ in range(player_count)]

    # array for hole numbers, will be used as a title
    holes: List[int] = [0] * hole_count

    # --- GATHER INPUT ------------------------------
    # datafile variable, open file
    with open(argv[3], 'r') as golf_data:
        # read in pars
        for i in range(hole_count):
            pars[i] = int(golf_data.readline().strip())

        # read in names
        for i in range(player_count):
            names[i] = int(golf_data.readline().strip())

        # read in strokes
        for i in range(hole_count):
            for j in range(player_count):
                results[i][j] = int(golf_data.readline().strip())

    # --- OUTPUT HOLES ------------------------------
    # hole number (title)
    print("\n       ", end="")
    for i in range(hole_count):
        print(f"{holes[i]:>3}", end="")
    print(" Scores\n")

    # print each player's number of strikes per hole
    for i in range(player_count):
        print_line(i,player_count,names,results)
    print()

    # --- CALCULATE SUMS ----------------------------
    # array for the total strokes of the game per player
    sums: List[int] = [0] * player_count

    for i in range(player_count):
        sums[i] = calculate_sum(i,hole_count,results)

    # --- FIND LOWEST SCORE -------------------------
    # set lowest to first total stroke array element, will iterate later to 
    # find true lowest
    lowest_score: int = sums[0]
    # index in array of winning player
    winning_player_index: int = 0

    for i in range(1,player_count):
        # find the lowest score, making note of index in array
        if sums[i] < lowest_score:
            lowest_score = sums[i]
            winning_player_index = i

    print(f"The winner is {names[winning_player_index]} with a score of {lowest_score}")


if __name__ == "__main__":
    main()