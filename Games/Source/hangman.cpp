/* HANGMAN
   William Wadsworth
   CSC1710
   Created: 11.3.2020
   Doctored: 10.16.2023
   ~/csc1710/prog3/
   Hangman

   todo: pick random word from array (array loaded from file with random words)

   2 args: ./exe <word bank file>
*/

#include <iostream>
#include <iomanip> // setw
#include <string> // length(), compare(), find()
#include <fstream> // file reading

#include <cstdlib> // rand(), srand()
#include <ctime> // time()
using namespace std;

// max amount of mistakes user can make
#define MAX_MISTAKES 100

// function prototypes
void userGuess(string, string&, string&, int&);
//char guess(string, int&);
bool checkMatch(char, string);

int main(int argc, char* argv[])
{
    // seed the random number generator -- ChatGPT
    srand(static_cast<unsigned>(time(nullptr)));
    
    // get filename from argv
    // need to convert from char* to string
    //string file = argv[1];
    ifstream inputFile;
    inputFile.open(argv[1]);

    // check if file name provided is valid. if not, throw error and quit
    if(!inputFile) {
        cerr << "error: file name provided (" << argv[1] << ") is invalid or not found. quitting...\n";
        exit(1);
    }

    // get number of words (should be first item in file)
    int wordBankNum;
    inputFile >> wordBankNum;
    // input validation

    // get random word from file
    // THIS SHOULD BE MAJORLY IMPROVED
    string word;
    for(int i=0; i<rand()%wordBankNum; i++)
        inputFile >> word;
    // close file, no longer needed
    inputFile.close();

    // word
    //string word = "experience";

    // mistakes user has made
    int incorrect = 0;
    // did the user finish
    //bool finish = false;
    // string of used letters, starts empty
    string used = "";

    // word with guessed letters, starts empty
    string evol = "";
    // initialize to blanks
    for(size_t i=0; i<word.length(); i++)
        evol += "_";
    
    // title
    //cout << setw(21) << "Hangman" << setw(21) << endl << " " << endl;
    cout << setw(15) << "Hangman" << endl << endl;

    // DEBUG
    //cout << "the word is " << word << " with length " << word.length() << endl;
    //cout << word << endl << evol << endl;

    // user must not be finished and have no more than 10 mistakes
    while (/*finish == false &&*/ incorrect < MAX_MISTAKES)
    {
        // print blanks for word
        // using size_t to avoid warning with -Wall
        for (size_t i=0; i<word.length(); i++) {
            //cout << setw(5);
            //cout << "_ ";
            cout << evol[i] << " ";
        }
        // extra space
        cout << endl << endl;

        // show user remaining mistakes
        cout << "Remaining guesses: " << (MAX_MISTAKES - incorrect) << endl;
        // show user guessed words
        cout << "Used letters: " << used << endl;
        //cout << "The word is: " << evol << endl;

        // add guessed letters to string
        //used += guess(used,incorrect);
        userGuess(word,evol,used,incorrect);

        // if the user correctly guessed the word, output to reflect
        if(incorrect < MAX_MISTAKES && evol.compare(word)==0) {
            cout << "\nCongratulations, you win! The word was " << word << endl;
            return 0;
            //finish = true;
        }

        // if user makes 10 mistakes, show word and quit program
        if (incorrect == MAX_MISTAKES) {
            cout << "\nYou lose. The word was " << word << endl;
            return 0;
            //finish = true;
        }
    }

    return 0;
}


void userGuess(string word, string& guessWord, string& used, int& incorrect)
{
    
    char guess;
    cout << "Guess a letter: ";
    cin >> guess;
    
    // check if the user has used this letter
    while(checkMatch(guess,used)) {
        // if user guesses letter again
        cout << "You already guessed " << guess << endl;
        cout << "Guess another letter: ";
        cin >> guess;
    }

    // if the letter is in the word, update guessWord
    if (checkMatch(guess, word)) {
        // update guessWord with correct guessed letters
        cout << guess << " is in the word" << endl;

        for (size_t i=0; i<word.length(); i++)
            if (word[i] == guess)
                guessWord[i] = guess;
    }
    // letter is not in word
    else {
        // add to mistake counter if incorrect
        cout << guess << " is not in the word" << endl;
        // update used words string/array
        used = used + guess + " ";
        // increment incorrect number of guesses
        incorrect++;
    }
}


bool checkMatch(char letter, string word)
{
    return (word.find(letter) != string::npos);
}


