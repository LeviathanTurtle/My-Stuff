# HANGMAN -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.3.2020
# Doctored: 10.16.2023
# Python-ized: 3.30.2024
# Updated 8.16.2024: function decomposition and PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program is hangman the game.
# 
# [USAGE]:
# python3 hangman.py <word bank file>

# --- IMPORTS ---------------------------------------------------------------------------
"""
#include <iostream>
#include <iomanip> // setw
#include <string> // length(), compare(), find()
#include <fstream> // file reading

#include <cstdlib> // rand(), srand()
#include <ctime> // time()
using namespace std;

#define MAX_MISTAKES 100
"""
from sys import argv, stderr
from random import seed, randint
from time import time
from typing import List

# max amount of mistakes user can make
MAX_MISTAKES: int = 10

DEBUG: bool = False

# --- FUNCTIONS -------------------------------------------------------------------------
# --- USER GUESS --------------------------------
"""
void userGuess(string, string&, string&, int&);
void userGuess(string word, string& guessWord, string& used, int& incorrect)
{
    char guess;
    cout << "Guess a letter: ";
    cin >> guess;
    
    while(checkMatch(guess,used)) {
        cout << "You already guessed " << guess << endl;
        cout << "Guess another letter: ";
        cin >> guess;
    }
    if (checkMatch(guess, word)) {
        cout << guess << " is in the word" << endl;

        for (size_t i=0; i<word.length(); i++)
            if (word[i] == guess)
                guessWord[i] = guess;
    }
    else {
        cout << guess << " is not in the word" << endl;
        used = used + guess + " ";
        incorrect++;
    }
}
"""
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
    """
    int main(int argc, char* argv[])
    {
        srand(static_cast<unsigned>(time(nullptr)));
        
        ifstream inputFile;
        inputFile.open(argv[1]);

        if(!inputFile) {
            cerr << "error: file name provided (" << argv[1] << ") is invalid or not found. quitting...\n";
            exit(1);
        }
        
        int wordBankNum;
        inputFile >> wordBankNum;

        string word;
        for(int i=0; i<rand()%wordBankNum; i++)
            inputFile >> word;

        inputFile.close();
    """
    # seed the random number generator
    seed(time())
    
    if len(argv) != 2:
        stderr.write("Usage: python hangman.py <wordlist_file>\n")
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
    """
        int incorrect = 0;

        string used = "";

        string evol = "";
        
        for(size_t i=0; i<word.length(); i++)
            evol += "_";
    """
    # mistakes user has made
    incorrect_guesses: int = 0
    # string of used letters, starts empty
    used_letters: List[str] = []
    # word with guessed letters, initialize to blanks
    guess_word: str = ["_"] * len(word)

    # --- TITLE ---------------------------------
    """
        cout << setw(15) << "Hangman" << endl << endl;
    """
    print(f"{'Hangman':>15}\n")

    # --- MAIN LOOP ---------------------------------------------------------------------
    # --- PRINT BLANKS --------------------------
    """
        while (/*finish == false &&*/ incorrect < MAX_MISTAKES)
        {
            for (size_t i=0; i<word.length(); i++)
                cout << evol[i] << " ";

            cout << endl << endl;
    """
    # user must not be finished and have no more than 10 mistakes
    while incorrect_guesses < MAX_MISTAKES:
        # print blanks for word
        """for i in range(len(word)):
            print(f"{evol[i]} ",end="")
        # extra space
        print()"""
        print(' '.join(guess_word))

    # --- SHOW REMAINING MISTAKES ---------------
        """
            cout << "Remaining guesses: " << (MAX_MISTAKES - incorrect) << endl;
            
            cout << "Used letters: " << used << endl;
        """
        # show user remaining mistakes
        print(f"\nRemaining guesses: {MAX_MISTAKES-incorrect_guesses}")
        # show user guessed words
        print(f"Used letters: {', '.join(used_letters)}")

    # --- ADD GUESSED LETTERS -------------------
        """
            userGuess(word,evol,used,incorrect);
        """
        # add guessed letters to string
        incorrect_guesses = user_guess(word,guess_word,used_letters,incorrect_guesses)

    # --- CHECK MISTAKES ----------------------------
        """
        if(incorrect < MAX_MISTAKES && evol.compare(word)==0) {
                cout << "\nCongratulations, you win! The word was " << word << endl;
                return 0;
            }
        NOTE: this was moved here to keep things consistent
        
        if (incorrect == MAX_MISTAKES) {
                cout << "\nYou lose. The word was " << word << endl;
                return 0;
            }
        }

        return 0;
    }
        """
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