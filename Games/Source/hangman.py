# HANGMAN -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.3.2020
# Doctored: 10.16.2023
# 
# Python-ized: 3.30.2024
# Updated 8.16.2024: function decomposition and PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program is hangman the game.
# 
# [USAGE]:
# python3 hangman.py <word bank file>

# --- IMPORTS ---------------------------------------------------------------------------
from sys import argv, stderr, exit
from random import seed, randint
from time import time
from typing import List

# max amount of mistakes user can make
MAX_MISTAKES: int = 10
DEBUG: bool = False


# --- FUNCTIONS -------------------------------------------------------------------------
# --- USER GUESS --------------------------------
def user_guess(word: str, guess_word: str, used_letters: str, incorrect_guesses: int) -> int:
    """Handle the user's letter guess and update the game state."""
    
    if DEBUG:
        print("Entering user_guess...")
    
    guess = input("\nGuess a letter: ").lower()
    
    # check if the user has used this letter
    while guess in used_letters:
        # if user guesses letter again
        print(f"You already guessed '{guess}'")
        guess = input("Guess another letter: ").lower()

    # if the letter is in the word, update guessWord
    if guess in word:
        # update guessWord with correct guessed letters
        print(f"'{guess}' is in the word")
        
        for i, letter in enumerate(word):
            if letter == guess:
                guess_word[i] = guess
        # update used letters
        used_letters.append(guess)
            
    # letter is not in word
    else:
        # add to mistake counter if incorrect
        print(f"'{guess}' is not in the word")
        # update used words string/array
        #used = used + guess + " "
        used_letters.append(guess)
        # increment incorrect number of guesses
        incorrect_guesses += 1
    
    if DEBUG:
        print("Exiting user_guess.")
    return incorrect_guesses


def main():
    # --- SETUP ---------------------------------
    # seed the random number generator
    seed(time())
    
    if len(argv) != 2:
        stderr.write("Usage: python3 hangman.py <wordlist_file>\n")
        exit(1)

    # get filename from argv
    try:
        with open(argv[1], 'r') as input_file:
            word_bank_num = int(input_file.readline().strip())
            for _ in range(randint(0, word_bank_num - 1)):
                word = input_file.readline().strip()
    except (FileNotFoundError, ValueError) as e:
        stderr.write(f"Error: {str(e)}\n")
        exit(1)

    # --- SET UP VARS ---------------------------
    # mistakes user has made
    incorrect_guesses: int = 0
    # string of used letters, starts empty
    used_letters: List[str] = []
    # word with guessed letters, initialize to blanks
    guess_word: str = ["_"] * len(word)

    # --- TITLE ---------------------------------
    print(f"{'Hangman':>15}\n")

    # --- MAIN LOOP ---------------------------------------------------------------------
    # --- PRINT BLANKS --------------------------
    # user must not be finished and have no more than 10 mistakes
    while incorrect_guesses < MAX_MISTAKES:
        # print blanks for word
        """for i in range(len(word)):
            print(f"{evol[i]} ",end="")
        # extra space
        print()"""
        print(' '.join(guess_word))

    # --- SHOW REMAINING MISTAKES ---------------
        # show user remaining mistakes
        print(f"\nRemaining guesses: {MAX_MISTAKES-incorrect_guesses}")
        # show user guessed words
        print(f"Used letters: {', '.join(used_letters)}")

    # --- ADD GUESSED LETTERS -------------------
        # add guessed letters to string
        incorrect_guesses = user_guess(word,guess_word,used_letters,incorrect_guesses)

    # --- CHECK MISTAKES ------------------------
        # if the user correctly guessed the word, output to reflect
        if ''.join(guess_word) == word:
            print(f"\nCongratulations, you win! The word was '{word}'")
            exit(0)
        # if user makes 10 mistakes, show word and quit program
        elif incorrect_guesses == MAX_MISTAKES:
            print(f"\nYou lose. The word was '{word}'")
            exit(0)


if __name__ == "__main__":
    main()