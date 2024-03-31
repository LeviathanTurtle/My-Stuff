# HANGMAN -- V.PY
# William Wadsworth
# CSC1710
# Created: 11.3.2020
# Doctored: 10.16.2023
# Python-ized: 3.30.2024
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
import sys
import random
import time

# max amount of mistakes user can make
MAX_MISTAKES = 10

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
def userGuess(word, guess_word, used, incorrect) -> int:
    guess = input("\nGuess a letter: ")
    
    # check if the user has used this letter
    #while checkMatch(guess,used):
    while guess in used:
        # if user guesses letter again
        print("You already guessed ", guess)
        guess = input("Guess another letter: ")

    # if the letter is in the word, update guessWord
    #if checkMatch(guess,word):
    if guess in word:
        # update guessWord with correct guessed letters
        print(guess,"is in the word")
        
        for i in range(len(word)):
            if(word[i] == guess):
                guess_word[i] = guess
        # update used letters
        used.append(guess)
            
    # letter is not in word
    else:
        # add to mistake counter if incorrect
        print(guess,"is not in the word")
        # update used words string/array
        #used = used + guess + " "
        used.append(guess)
        # increment incorrect number of guesses
        incorrect += 1
    
    return incorrect

# --- CHECK MATCH -------------------------------
"""
bool checkMatch(char, string);
bool checkMatch(char letter, string word)
{
    return (word.find(letter) != string::npos);
}
"""
def checkMatch(letter, word) -> bool:
    return letter in word

# --- MAIN ------------------------------------------------------------------------------
# --- SETUP -------------------------------------
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
"""
# seed the random number generator -- ChatGPT
random.seed(time.time())

# get filename from argv
with open(sys.argv[1],'r') as input_file:
    # check if file name provided is valid. if not, throw error and quit
    if not input_file:
        sys.stderr.write(f"error: file name provided ({sys.argv[1]}) is invalid or not found.")
        exit(1)
    
# --- READ FILE ---------------------------------
    """
    int wordBankNum;
    inputFile >> wordBankNum;

    string word;
    for(int i=0; i<rand()%wordBankNum; i++)
        inputFile >> word;

    inputFile.close();
    """
    # get number of words (should be first item in file)
    word_bank_num = int(input_file.readline())
    
    # get random word from file
    for _ in range(random.randint(0, word_bank_num-1)):
        word = input_file.readline().strip()

# --- SET UP VARS -------------------------------
"""
    int incorrect = 0;

    string used = "";

    string evol = "";
    
    for(size_t i=0; i<word.length(); i++)
        evol += "_";
"""
# mistakes user has made
#incorrect = 0
incorrect = 0
# string of used letters, starts empty
#used = ""
used = []
# word with guessed letters, starts empty
#evol = ""
# initialize to blanks
#for _ in range(len(word)):
#    evol += "_"
evol = ["_"] * len(word)

# --- TITLE -------------------------------------
"""
    cout << setw(15) << "Hangman" << endl << endl;
"""
print(f"{'Hangman':>15}\n")

# --- MAIN LOOP -------------------------------------------------------------------------
# --- PRINT BLANKS ------------------------------
"""
    while (/*finish == false &&*/ incorrect < MAX_MISTAKES)
    {
        for (size_t i=0; i<word.length(); i++)
            cout << evol[i] << " ";

        cout << endl << endl;
"""
# user must not be finished and have no more than 10 mistakes
while incorrect < MAX_MISTAKES:
    # print blanks for word
    for i in range(len(word)):
        print(f"{evol[i]} ",end="")
    # extra space
    print()

# --- SHOW REMAINING MISTAKES -------------------
    """
        cout << "Remaining guesses: " << (MAX_MISTAKES - incorrect) << endl;
        
        cout << "Used letters: " << used << endl;
    """
    # show user remaining mistakes
    print("Remaining guesses:",MAX_MISTAKES-incorrect)
    # show user guessed words
    print("Used letters:",used)

# --- ADD GUESSED LETTERS -----------------------
    """
        userGuess(word,evol,used,incorrect);
    """
    # add guessed letters to string
    incorrect = userGuess(word,evol,used,incorrect)

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
    evol_str = ''.join(evol)
    # if the user correctly guessed the word, output to reflect
    if incorrect < MAX_MISTAKES and evol_str == word:
        print("\nCongratulations, you win! The word was",word)
        exit(0)
    # if user makes 10 mistakes, show word and quit program
    elif incorrect == MAX_MISTAKES:
        print("\nYou lose. The word was",word)
        exit(0)