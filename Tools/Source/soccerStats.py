# SOCCER PLAYER STATISTICS -- V.PY
# William Wadsworth
# CSC1710
# Created: 1.12.2021
# Doctored: 11.2.2023
# 
# Python-ized: 3.30.2024
# Updated 8.17.2024: function decomposition and PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program will store a soccer player's name, position, number of games played, total goals
# made, number of shots taken, and total number of minutes played. It will then compute the shot
# percentage and output the information in a formatted table.
# 
# [USAGE]:
# python3 soccerStats <input file>
# 
# [INPUT FILE STRUCTURE]:
# first_name last_name position games goals shots minutes
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed a full execution
# 
# 1 - CLI args used incorrectly
# 
# 2 - file unable to be opened or created

# --- IMPORTS ---------------------------------------------------------------------------
from sys import argv, stderr, exit
from dataclasses import dataclass
from typing import List

MAX_SIZE: int = 30
DEBUG: bool = False


# --- OBJECTS ---------------------------------------------------------------------------
@dataclass
class Player:
    name: str = ""
    position: str = ""
    games: int = 0
    goals: int = 0
    shots: int = 0
    minutes: int = 0


# --- FUNCTIONS -------------------------------------------------------------------------
# --- LOAD DATA ---------------------------------
# Pre-condition: the player array references the array that will be loaded
#                with the player data
# Post-condition: the player array will be loaded with the data found in
#                 the data file, but not exceeding the max of 30
# Assumption: if the player's name can be read, assume that the position,
#             games played, goals, shots, and minutes played follows.
def load_data(filename: str, players: List[Player], start_count: int) -> int:
    """Loads player data from a file into the players array."""
    
    if DEBUG:
        print("Entering load_data...")
        
    # open file
    try:
        with open(filename, 'r') as file:
            # get the players + stats
            for count in range(start_count, MAX_SIZE):
                data = file.readline().split()
                if not data:
                    break
                
                players[count].name = data[0]
                players[count].position = data[1]
                players[count].games = int(data[2])
                players[count].goals = int(data[3])
                players[count].shots = int(data[4])
                players[count].minutes = int(data[5])
    except FileNotFoundError:
        stderr.write(f"Error: File '{filename}' not found")
        exit(2)
    
    if DEBUG:
        print("Exiting load_data.")
    return count


# --- PRINT DATA --------------------------------
# Pre-condition: the player array (team[]) is loaded with player data for n
#                players
# Post-condition: the player array will be printed, no changes made
def print_data(players: List[Player], count: int) -> None:
    """Prints the soccer statistics of the players."""
    
    if DEBUG:
        print("Entering print_data...")
        
    # titles
    print(f"{'HPU Soccer Stats':>10}")
    print(f"{'Name':<11}{'Position':<12}{'GP':>4}{'G':>4}{'SH':>6}{'Mins':>7}{'Shot %':>8}")
    
    # individual data
    for i in range(count):
        shot_percentage = (f"{players[i].goals*100 / players[i].shots:>7.2f}%" if players[i].shots != 0 else "0.0%")
        
        print(f"{players[i].name:<11}{players[i].position:<12}"
              f"{players[i].games:>4}{players[i].goals:>4}"
              f"{players[i].shots:>6}{players[i].minutes:>7}"
              f"{shot_percentage:>8}")
    
    if DEBUG:
        print("Exiting print_data.")


def main():
    # --- CHECK CLI ARGS ------------------------
    # check that CLI args are used correctly
    if len(argv) != 2:
        stderr.write("Usage: python3 soccerStats.py <input file>")
        exit(1)

    # --- LOAD AND PRINT ------------------------
    # database array
    players: List[Player] = [Player() for _ in range(MAX_SIZE)]

    # load array, keeping array size
    player_count = load_data(argv[1],players,0)
    print_data(players,player_count)

    print(f"Player count: {player_count}")


if __name__ == "__main__":
    main()